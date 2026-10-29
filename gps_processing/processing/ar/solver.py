"""Static/kinematic experimental GNSS AR filter (GPS L1/L2, Galileo E1/E5a). Fixed estimates
never feed the float filter.

Each satellite system has its own reference satellite (ambiguity datum), its own receiver
clocks/biases (DCM: code, phase, MW; IPC: code, phase; SDBS: removed by between-satellite
differences inside each system) and its own wavelengths; the WL and N1 integers of both
systems are searched together, since every candidate is a within-system difference.

``estimator='sequential'`` reproduces the v2.0 static filter. The EKF estimators add
process noise, a zenith wet delay state, Saastamoinen/relativity corrections and robust
(IGG-III) updates; ``integer_method`` selects LAMBDA/ILS, integer bootstrapping or TCAR/CIR-style cascade rounding
(all with partial AR except 'ils'), and
``ar_validation`` may use the learned acceptance test.
"""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from . import sdbs,dcm,ipc
from .models import LAMBDA_W,LAMBDA_N,BETA,combination_covariance
from .integer import evaluate,condition_linear
from gps_processing.processing.geometry import elevation
from gps_processing.processing.cycle_slip import SlipDetector
from gps_processing.processing.atmosphere import Atmosphere
from gps_processing.processing.estimation import ExtendedKalmanFilter,FilterOptions
from gps_processing.constants import C
from gps_processing import gnss

def ar_settings(config):
    return SimpleNamespace(method=getattr(config,'integer_method','ils'),ratio=config.ar_ratio,max_std=config.ar_max_std,
        success=getattr(config,'ar_success_rate',.999),validation=getattr(config,'ar_validation','ratio'),
        threshold=getattr(config,'ml_threshold',.99))

def fix_stage(state,cov,indices,settings,validator,stage):
    """One integer stage (WL or N1). Returns fixed state/covariance or None and a reason."""
    if settings.method=='ils' and np.sqrt(np.maximum(0,np.diag(cov)[indices])).max()>=settings.max_std:
        return None,None,f'{stage}_precision_gate'
    result=evaluate(state[indices],cov[np.ix_(indices,indices)],settings.method,settings.success)
    # With the learned test the success rate is one of its inputs, so the hard Ps gate is left
    # to it; PAR still chooses the subset with ``settings.success``.
    if settings.method!='ils' and 'ml' not in settings.validation and result['success']<settings.success:
        return None,result,f'{stage}_success_rate_gate'
    if 'ratio' in settings.validation and result['ratio']<settings.ratio:
        return None,result,f'{stage}_ratio_rejected'
    if 'ml' in settings.validation:
        result['probability']=validator.probability(result,stage)
        if result['probability']<settings.threshold:
            return None,result,f'{stage}_ml_rejected'
    fixed,q=condition_linear(state,cov,indices,result['combinations'],result['integers'])
    return (fixed,q),result,''

def system_satellites(system):
    info=gnss.SYSTEMS[system]
    return [info['offset']+prn for prn in range(1,info['count']+1)]

def amb(first,sat):
    """State index of the N1 ambiguity of ``sat`` (NW follows it)."""
    return first+2*gnss.slot(sat)

def redatum(kf,first,new):
    """Re-reference the (N1, NW) pairs of ``new``'s system to satellite ``new``: x_j -= x_new."""
    t=np.eye(len(kf.x)); k=amb(first,new)
    for sat in system_satellites(gnss.system_of(new)):
        j=amb(first,sat)
        t[j,k]-=1; t[j+1,k+1]-=1
    kf.x=t@kf.x; kf.P=t@kf.P@t.T

def solve(obs,satellites,approximate_xyz,config,variance_model=None,validator=None,atmosphere=None):
    options=FilterOptions.from_config(config); settings=ar_settings(config)
    if 'ml' in settings.validation and validator is None:
        from gps_processing.processing.ml.ar_validation import default_validator
        validator=default_validator()
    method=config.technique.split('/')[-1]
    model={'SDBS':sdbs,'DCM':dcm,'IPC':ipc}[method]
    clocks={'SDBS':[],'DCM':[3,4,5],'IPC':[3,4]}[method]
    resolvers={sys:ipc.WideLaneResolver(config.ar_min_epochs,lambda_w=gnss.constants(sys)['lambda_w']) for sys in gnss.SYSTEMS} if method=='IPC' else None
    resolver=resolvers['G'] if resolvers else None
    wide_lane={}
    columns=['epok','sat','x','y','z']+(['relativity_m'] if options.physical_corrections and 'relativity_m' in satellites else [])
    data=obs.merge(satellites[columns],on=['epok','sat'],validate='one_to_one')
    base=3+len(clocks)
    tropo=base if options.estimate_troposphere else None
    first=base+(tropo is not None)
    galileo=first+2*gnss.SLOTS                       # Galileo receiver clocks/biases follow the ambiguities
    clock_of={'G':clocks,'E':[galileo+k for k in range(len(clocks))]}
    dim=galileo+len(clocks)
    atmosphere=atmosphere or Atmosphere(approximate_xyz,physical=options.physical_corrections)
    state=np.zeros(dim); state[:3]=approximate_xyz
    if tropo is not None: state[tropo]=atmosphere.initial_zwd
    kf=ExtendedKalmanFilter(state,np.diag([1e4]*3+[1e10]*(base-3)+([atmosphere.initial_zwd_variance] if tropo is not None else [])
                                         +[1e6]*(2*gnss.SLOTS)+[1e10]*len(clocks)))
    detector=SlipDetector(gf_jump=getattr(config,'gf_slip_m',.05) if options.is_ekf else None)
    references={}; age={}; last_epoch=None
    positions=[]; ambiguities=[]; diagnostics=[]; correction=np.zeros(dim)
    for epoch,frame in data.groupby('epok',sort=True):
        frame=frame.copy()
        frame['elevation']=elevation(kf.x[:3],frame[['x','y','z']].to_numpy())
        frame=frame[frame.elevation>=config.elevation_mask].sort_values('sat').reset_index(drop=True)
        if len(frame)<5:
            diagnostics.append(dict(epok=epoch,status='skipped',reason='fewer_than_5_satellites',satellites=len(frame)))
            continue
        kf.predict(options.process_noise(dim,epoch-last_epoch if last_epoch is not None else 0,tropo))
        last_epoch=epoch
        ids=frame.sat.to_numpy(dtype=int)
        slipped=detector.check(epoch,frame)
        for sat in slipped:
            kf.reset([amb(first,sat),amb(first,sat)+1],0.0,1e6)
            age[sat]=0
            wide_lane.pop(sat,None)
            if resolvers is not None: resolvers[gnss.system_of(sat)].reset(sat)
        present=gnss.systems_of(ids)
        for system in gnss.SYSTEMS:
            mine=ids[present==system]
            reference=references.get(system)
            if not len(mine):
                continue
            if reference not in mine or reference in slipped:
                sub_frame=frame[frame.sat.isin(mine)]
                new=int(sub_frame.loc[sub_frame.elevation.idxmax(),'sat'])
                own=[s for s in wide_lane if gnss.system_of(s)==system]
                if reference is not None and age.get(new,0)>0 and new not in slipped:
                    # The datum change is an integer shift of the system's ambiguities (absorbed by its
                    # receiver clocks, Shi & Gao 2014): N_j <- N_j - N_new keeps converged and fixed arcs.
                    redatum(kf,first,new)
                    if new in wide_lane:
                        shift=wide_lane[new]
                        for s in own: wide_lane[s]-=shift
                        wide_lane.pop(new)
                    elif own:
                        # IPC: the new reference has no integer WL yet; re-resolve against it.
                        for s in own: kf.reset(amb(first,s)+1,0.0,1e6); wide_lane.pop(s)
                else:
                    every=system_satellites(system)
                    kf.reset(np.ravel([[amb(first,s),amb(first,s)+1] for s in every]),0.0,1e6)
                    for s in every: age.pop(s,None); wide_lane.pop(s,None)
                references[system]=new
            kf.reset([amb(first,references[system]),amb(first,references[system])+1],0.0,0.0)
        reference=references.get('G',references.get('E'))
        for sat in ids: age[sat]=age.get(sat,0)+1
        if clocks:
            kf.reset(clocks+clock_of['E'],0.0,1e10)
        frame['arc_epochs']=[age[s] for s in ids]
        n=len(frame); noise=np.zeros((3*n,3*n))
        y=np.empty(3*n)
        sigmas=variance_model.sigmas(frame) if variance_model is not None else None
        for i,row in enumerate(frame.itertuples()):
            y[3*i:3*i+3]=[row.Pc+row.code_m,row.Lc+row.phase_m,row.A4+row.mw_m]
            if resolvers is not None: resolvers[gnss.system_of(row.sat)].add(int(row.sat),y[3*i+2])
            f1,f2=gnss.frequencies(int(row.sat)); f1,f2=float(f1[0]),float(f2[0])
            if sigmas is None:
                noise[3*i:3*i+3,3*i:3*i+3]=combination_covariance(max(.1,np.sin(np.radians(row.elevation))),options.code_sigma,options.phase_sigma,f1,f2)
            else:
                noise[3*i:3*i+3,3*i:3*i+3]=combination_covariance(1.0,sigmas[0][i],sigmas[1][i],f1,f2)
        coords=frame[['x','y','z']].to_numpy()
        delays=atmosphere.slant(epoch,kf.x[:3],coords,frame.elevation.to_numpy(),frame.sat.to_numpy()); wet=delays.wet
        # Satellite-specific delays common to code and phase (MW is geometry free).
        common=delays.common.copy()
        if options.physical_corrections and 'relativity_m' in frame:
            common-=frame.relativity_m.to_numpy()
        position={s:i for i,s in enumerate(ids)}
        ref_pos=np.array([position[references[gnss.system_of(s)]] for s in ids])   # per-satellite datum position
        if resolvers is not None:
            # IPC wide lane: rounded with the satellite-averaged receiver bias of each system, then held as known.
            for system,res in resolvers.items():
                mine=ids[present==system]
                if not len(mine): continue
                undifferenced=res.resolve(mine); ref=references[system]
                if ref in undifferenced:
                    for sat,value in undifferenced.items():
                        if sat!=ref and sat not in wide_lane:
                            wide_lane[sat]=value-undifferenced[ref]
                            kf.reset(amb(first,sat)+1,float(wide_lane[sat]),1e-10)
        is_reference=ref_pos==np.arange(n)
        groups={'SDBS':np.repeat([i for i in range(n) if not is_reference[i]],3),'DCM':np.repeat(np.arange(n),3),
                'IPC':np.repeat(np.arange(n),2)}[method]
        wave={s:gnss.constants(gnss.system_of(s)) for s in ids}
        def equations(current,inflation=np.ones(n)):
            delta=current[:3]-coords; ranges=np.linalg.norm(delta,axis=1)
            h=np.zeros((3*n,dim)); predicted=np.zeros(3*n)
            for i,sat in enumerate(ids):
                k=amb(first,sat); w=wave[sat]
                h[3*i,:3]=delta[i]/ranges[i]; h[3*i+1,:3]=h[3*i,:3]
                h[3*i+1,k:k+2]=[w['lambda_n'],w['beta']]
                h[3*i+2,k+1]=w['lambda_w']
                predicted[3*i:3*i+3]=[ranges[i]+common[i]+delays.code[i],ranges[i]+common[i]+delays.phase[i]+w['lambda_n']*current[k]+w['beta']*current[k+1],w['lambda_w']*current[k+1]]
                if tropo is not None:
                    h[3*i:3*i+2,tropo]=wet[i]
                    predicted[3*i:3*i+2]+=wet[i]*current[tropo]
                for row_offset,clock in enumerate(clock_of[gnss.system_of(sat)]):
                    h[3*i+row_offset,clock]=1
                    predicted[3*i+row_offset]+=current[clock]
            scaled=noise*np.repeat(inflation,3)[:,None]**.5*np.repeat(inflation,3)[None,:]**.5
            return (*model.transform(y,predicted,h,scaled,ref_pos),groups)
        prior=kf.x.copy()
        report=kf.update(equations,n,robust=options.robust,adaptive=options.adaptive,adaptive_indices=[0,1,2])
        state,cov=kf.x,kf.P
        correction=state-prior
        fixed=state.copy(); status='float'; reason='arc_or_precision_gate'
        wl=nl=None
        datum=lambda s:references[gnss.system_of(s)]
        candidates=[int(s) for s in ids if s!=datum(s) and age[s]>=config.ar_min_epochs and age[datum(s)]>=config.ar_min_epochs
                    and (resolvers is None or s in wide_lane)]
        candidates.sort(key=lambda s:cov[amb(first,s),amb(first,s)])
        chosen=candidates[:4] if settings.method=='ils' else candidates
        w=[amb(first,s)+1 for s in chosen]; narrow=[i-1 for i in w]
        if len(chosen)>=2:
            try:
                stage,wl,reason=fix_stage(state,cov,w,settings,validator,'wl')
                if stage is not None:
                    stage,nl,reason=fix_stage(*stage,narrow,settings,validator,'n1')
                    if stage is not None:
                        candidate=stage[0]
                        measured,predicted,_,rr,_=equations(candidate)
                        residual=measured-predicted
                        normalized=float(residual@np.linalg.solve(rr,residual)/len(residual))
                        if normalized<=9:
                            fixed=candidate; reason='wl_n1_accepted'
                            full=wl['dimension']==nl['dimension']==len(candidates)
                            status='fixed' if full else 'partial_fixed'
                        else: reason='residual_gate'
            except (RuntimeError,np.linalg.LinAlgError):
                reason='search_incomplete_or_singular'
        measured,predicted,_,_,_=equations(fixed)
        residual=measured-predicted
        rms=float(np.sqrt(np.mean(residual**2)))
        sigma=np.sqrt(np.diag(cov)[:3])
        positions.append(dict(epok=epoch,x=fixed[0],y=fixed[1],z=fixed[2],
            float_x=state[0],float_y=state[1],float_z=state[2],sigma_x=sigma[0],sigma_y=sigma[1],sigma_z=sigma[2],
            receiver_clock_s=state[3]/C if clocks else np.nan,
            receiver_phase_clock_s=state[4]/C if clocks else np.nan,
            receiver_mw_bias_m=state[5] if method=='DCM' else (resolver.receiver_bias*LAMBDA_W if resolver is not None else np.nan),
            reference_sat_E=references.get('E',np.nan) if 'G' in references else np.nan,
            gps_satellites=int((present=='G').sum()),galileo_satellites=int((present=='E').sum()),
            zwd_m=delays.zwd_apriori+state[tropo] if tropo is not None else (delays.zwd_apriori if atmosphere.tro is not None else np.nan),
            stec_tecu=float(np.nanmean(delays.stec)) if np.isfinite(delays.stec).any() else np.nan,
            satellites=n,status=status,reference_sat=reference,residual_rms_m=rms,nis=report.nis,alpha=report.alpha,
            downweighted=report.downweighted))
        def integer_value(value):
            return value if status!='float' and abs(value-np.rint(value))<1e-6 else np.nan
        for sat in ids:
            k=amb(first,sat)
            ambiguities.append(dict(epok=epoch,sat=sat,reference_sat=datum(sat),N1_float=state[k],NW_float=state[k+1],
                N1_fixed=integer_value(fixed[k]) if sat!=datum(sat) else np.nan,
                NW_fixed=integer_value(fixed[k+1]) if sat!=datum(sat) else np.nan))
        info=lambda r,key:r[key] if r else np.nan
        diagnostics.append(dict(epok=epoch,status=status,reason=reason,satellites=n,reference_sat=reference,
            wl_ratio=info(wl,'ratio'),n1_ratio=info(nl,'ratio'),wl_success=info(wl,'success'),n1_success=info(nl,'success'),
            wl_success_upper=info(wl,'success_upper'),n1_success_upper=info(nl,'success_upper'),wl_adop=info(wl,'adop'),n1_adop=info(nl,'adop'),
            wl_probability=wl.get('probability',np.nan) if wl else np.nan,n1_probability=nl.get('probability',np.nan) if nl else np.nan,
            candidate_pairs=len(chosen),fixed_pairs=nl['dimension'] if status!='float' else 0,
            wl_known=len(wide_lane) if resolvers is not None else np.nan,
            reset_satellites=','.join(map(str,slipped)),residual_rms_m=rms,nis=report.nis,downweighted=report.downweighted))
    if not positions: raise ValueError('PPP-AR için çözülebilir epok yok.')
    return pd.DataFrame(positions),pd.DataFrame(ambiguities),pd.DataFrame(diagnostics),kf.P,correction
