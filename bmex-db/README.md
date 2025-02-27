# bmex-db
Location of database management for BMEX projects.

# Workflow
When updating the database, we distinguish between two types of updates: majors and minors. Versions can be tracked with representations like vM.m (major of M and minor of m, eg. v2.3). The update script will be used to track all version info in ./db_versions.csv and update the db, ./bmex_masses.h5, with the utils/build_db.py script.

## Majors 
Majors represent large changes to the database. This may include adding a new dataset, reformatting existing datasets, adding one or more quantities to a dataset, etc. To indicate a major, use the "-m" flag with the update script.

./update_db.py <description of update> -m

This command will record the description and the new version (eg. v2.3 -> v3.1) in ./db_versions.csv. It will also rebuild the entire db using csvs in ./model_csvs and an added measurements in ./db_versions.csv.

## Minors
Minors represent small changes to the database. This will typically be the addition of a single measurement. 

./update_db.py <description of update> -v <value> -u <uncertainty> -z <proton number> -n <neutron number> -d <dataset> -q <quantity> -r <reference> -e

Note: the "-e" flag indicates the added value is estimated. Typically, we will add non-estimated binding energies to the experimental dataest, so the more command will be:

./update_db.py <description of update> -v <value> -u <uncertainty> -z <proton number> -n <neutron number> -r <reference>

These commands will increase the current minor version (eg. v2.3 -> v2.4) and store information about the added measurement. It will only rebuild the dataset of the added value. 