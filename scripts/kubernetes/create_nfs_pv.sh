if [ $# -lt 8 ]; then
    echo "usage:
         $0: <pv-name> <nfs-ip> <path> <storage> <accessModes> <reclaimPolicy> <lables_key> <lables_value>
         eg: $0 stolon-db-pv1 192.168.100.103 /home/rafiki/database/db1 100Gi ReadWriteOnce Retain"
    exit 1
fi

TMP_NFS_PV_YAML=scripts/kubernetes/tmp-nfs-pv.yaml
cp scripts/kubernetes/nfs-pv.yaml.template $TMP_NFS_PV_YAML
sed -ri "s/PV_NAME/$1/g" $TMP_NFS_PV_YAML
sed -ri "s/PV_IP/$2/g" $TMP_NFS_PV_YAML
sed -ri "s#PV_PATH#$3/#" $TMP_NFS_PV_YAML
sed -ri "s/PV_STORAGE/$4/g" $TMP_NFS_PV_YAML
sed -ri "s/PV_ACCESS_MODES/$5/g" $TMP_NFS_PV_YAML
sed -ri "s/PV_RECLAIN_POLICY/$6/g" $TMP_NFS_PV_YAML
sed -ri "s/LABLES_KEYS_0/$7/g" $TMP_NFS_PV_YAML
sed -ri "s/LABLES_VALUES_0/$8/g" $TMP_NFS_PV_YAML

kubectl create -f $TMP_NFS_PV_YAML
rm -f $TMP_NFS_PV_YAML
