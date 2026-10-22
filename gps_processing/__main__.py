import sys
if '--gui' in sys.argv:
    from gps_processing.ui.app import main
else:
    from gps_processing.cli import main
main()
