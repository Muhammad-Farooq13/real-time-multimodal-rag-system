from __future__ import annotations

from ray import serve

from app.main import app


@serve.deployment(num_replicas=2)
@serve.ingress(app)
class RagApiDeployment:
    pass


deployment = RagApiDeployment.bind()
