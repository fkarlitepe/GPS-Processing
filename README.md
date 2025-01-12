
---------------
GPS Precise Point Positioning Processing Software
---------------

Precise Point Positioning (PPP) uses code and phase signals to calculate the distance between the satellite and the receiver. 

In PPP, the functional model of the code-phase measure is affected by some error sources such as satellite position information, satellite-receiver clock and integer ambiguity


> - **karlitepe_processing.py** file should be run.

> - **produce_sp3.py**
  .sp3 extension data from the day before, the day after and the same day are processed. Data with .sp3 extension is needed


>- **observation_rinex3.py**
 C1/C2/P1/P2 code measurements and L1/L2/L5 phase measurements of GPS satellites are processed from the observation file.
 Only suitable for observation files in rinex3 format. Data with .rnx3 extension is needed


> - **sat_regression.py**
Executes the satellite orbit regression process.


> - **sat_ele.py**
Satellite altitude is calculated

> - **PPP_process.py**
Ionosphere Free Least Squares Solution is performed.
