#!/bin/bash

for i in {1..50};
do
    /opt/mssql-tools/bin/sqlcmd -S localhost -U ${USER_ID} -P ${SA_PASSWORD} -i create_db.sql
    if [ $? -eq 0 ]
    then
        echo "setup.sql completed"
        break
    else
        echo "not ready yet..."
        sleep 1
    fi
done

# add new group colum with data group
data_groups=("train" "val" "test")
for data_group in ${data_groups[@]}
do
  awk "BEGIN{ FS = OFS = \";\" } { print \$0, (NR==1? \"data group\" : \"${data_group}\") }" /data/${data_group}.csv > /data/${data_group}_with_group.csv
  /opt/mssql-tools/bin/bcp WineQuality.dbo.Wines in "/data/${data_group}_with_group.csv" -c -t';' -F2 -S localhost -U ${USER_ID} -P ${SA_PASSWORD}
done
