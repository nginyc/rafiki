import os
import pandas as pd

from clipper_admin import ClipperConnection, DockerContainerManager
from clipper_admin.deployers import python as python_deployer


class ClipperAdapter:
    def __init__(self):
        self._clipper_conn = ClipperConnection(DockerContainerManager())

    def start(self):
        self._clipper_conn.start_clipper()

    def stop(self):
        self._clipper_conn.stop_all()

    def deploy_model(self, model_id, deployment_name):
        self._clipper_conn.connect()

        def predict(xs):
            return [str(x) for x in xs]

        python_deployer.deploy_python_closure(
            self._clipper_conn,
            name=deployment_name,
            input_type='bytes',
            func=predict,
            version=1
        )

    def create_app(self, name, slo_micros, model_deployment_names):
        self._clipper_conn.register_application(
            name=name,
            input_type='bytes',
            slo_micros=slo_micros,
            default_output=''
        )

        for deployment_name in model_deployment_names:
            self._clipper_conn.link_model_to_app(
                app_name=name,
                model_name=deployment_name
            )


    def delete_app(self, name):
        self._clipper_conn.delete_application(
            name=name
        )
