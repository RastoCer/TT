 python "$env:SUMO_HOME\tools\detector\edgeDataFromFlow.py" `
   -d .\zimak_detectors_definition.xml `
   -f .\zimak_flow_data.csv `
   --id-column Detector `
   --time-column Time `
   --time-scale 60 `
   -i 60 `
   -q flow `
   -v `
   -o .\edge_counts\zimak_edgedata.xml


python "$env:SUMO_HOME\tools\randomTrips.py" `
  -n ..\sumo_network\osm.net.xml.gz `
  -o .\candidates.trips.xml `
  -r .\candidates.rou.xml `
  -b 0 -e 86400 `
  --prefix cand `
  --seed 42

python "$env:SUMO_HOME\tools\routeSampler.py" `
  -r .\candidates.rou.xml `
  --edgedata-files .\edge_counts\zimak_edgedata.xml `
  -o .\sampled.rou.xml `
  --edgedata-attribute flow `
  --write-flows number `
  --optimize full `
  --attributes "departLane=`"best`" departPos=`"base`" departSpeed=`"max`""