if [ $# -lt 5 ]; then
    echo "usage:
         $0: <pvc-name> <storage> <accessModes> <lables_key> <lables_value>
         eg: $0 database-stolon-keeper-0 100Gi ReadWriteOnce pv database-pv-0"
    exit 1
fi

TMP_NFS_PVC_YAML=scripts/kubernetes/tmp-nfs-pvc.yaml
cp scripts/kubernetes/nfs-pvc.yaml.template $TMP_NFS_PVC_YAML
sed -ri "s/PVC_NAME/$1/g" $TMP_NFS_PVC_YAML
sed -ri "s/PV_STORAGE/$2/g" $TMP_NFS_PVC_YAML
sed -ri "s/PV_ACCESS_MODES/$3/g" $TMP_NFS_PVC_YAML
sed -ri "s/LABLES_KEYS_0/$4/g" $TMP_NFS_PVC_YAML
sed -ri "s/LABLES_VALUES_0/$5/g" $TMP_NFS_PVC_YAML

kubectl create -f $TMP_NFS_PVC_YAML
rm -f $TMP_NFS_PVC_YAML
