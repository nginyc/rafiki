STOLON_PATH=scripts/kubernetes/stolon
cp -f $STOLON_PATH/secret.yaml.template $STOLON_PATH/secret.yaml
sed -ri "s/STOLON_PASSWD/$POSTGRES_STOLON_PASSWD/g" $STOLON_PATH/secret.yaml

# replace config for stolon keeper
cp -f $STOLON_PATH/stolon-keeper.yaml.template         $STOLON_PATH/stolon-keeper.yaml
sed -ri "s#RAFIKI_IMAGE_STOLON#$RAFIKI_IMAGE_STOLON#"  $STOLON_PATH/stolon-keeper.yaml
sed -ri "s/POSTGRES_PORT/$POSTGRES_PORT/g"             $STOLON_PATH/stolon-keeper.yaml

# replace config for stolon proxy
cp -f $STOLON_PATH/stolon-proxy.yaml.template          $STOLON_PATH/stolon-proxy.yaml
sed -ri "s#RAFIKI_IMAGE_STOLON#$RAFIKI_IMAGE_STOLON#"  $STOLON_PATH/stolon-proxy.yaml
sed -ri "s/POSTGRES_PORT/$POSTGRES_PORT/g"             $STOLON_PATH/stolon-proxy.yaml

# replace config for stolon sentinel
cp -f $STOLON_PATH/stolon-sentinel.yaml.template       $STOLON_PATH/stolon-sentinel.yaml
sed -ri "s#RAFIKI_IMAGE_STOLON#$RAFIKI_IMAGE_STOLON#"  $STOLON_PATH/stolon-sentinel.yaml

# replace config for stolon proxy service
cp -f $STOLON_PATH/stolon-proxy-service.yaml.template  $STOLON_PATH/stolon-proxy-service.yaml
sed -ri "s/POSTGRES_EXT_PORT/$POSTGRES_EXT_PORT/g"     $STOLON_PATH/stolon-proxy-service.yaml
sed -ri "s/POSTGRES_PORT/$POSTGRES_PORT/g"             $STOLON_PATH/stolon-proxy-service.yaml
