# TMS MEP Detection and Analysis

Automated detection and analysis of Motor Evoked Potentials (MEPs) from TMS experiments in non-human primates.

## Installation

```bash
git clone https://github.com/yourusername/tms_detection
cd tms_detection
pip install -r requirements.txt
```

1.
```bash
python detection.py /Users/cameroncronheimer/Desktop/tms_detection/Nov5_Olive

python epoch.py /Users/cameroncronheimer/Desktop/tms_detection/Nov5_Olive

```
2. 
```bash
python analysis.py /Users/cameroncronheimer/Desktop/tms_detection/Nov5_Olive
```



```mermaid
flowchart LR
    subgraph Cluster["Kubernetes Cluster"]
        subgraph spark["Namespace: spark"]
            master["Spark Master Deployment"]
            workers["Spark Worker Deployment"]
            sparkCfg["ConfigMap: spark-config"]
            sparkSec["Secret: spark-db-creds"]

            master --> sparkCfg
            master --> sparkSec
            workers --> sparkCfg
            workers --> sparkSec
        end

        subgraph nifi["Namespace: nifi"]
            nifiPod["NiFi StatefulSet"]
            nifiCfg["ConfigMap: nifi-config"]
            nifiSec["Secret: nifi-ldap-creds"]

            nifiPod --> nifiCfg
            nifiPod --> nifiSec
        end

        subgraph mgmt["Namespace: management"]
            ui["UI Deployment"]
            backend["Backend Deployment"]
            mgmtCfg["ConfigMap: mgmt-config"]
            mgmtSec["Secret: mgmt-api-creds"]

            ui --> mgmtCfg
            backend --> mgmtCfg
            backend --> mgmtSec
        end
    end

    nfs["NFS Server"]:::storage
    nfs --> master
    nfs --> workers
    nfs --> nifiPod
    nfs --> backend

    classDef storage stroke-dasharray: 5 5;


```