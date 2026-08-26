import pytest

import coribench.isaac_session as isaac_session


@pytest.fixture(autouse=True)
def clean_session():
    isaac_session._SHARED_ISAAC_RESOURCES.clear()
    isaac_session._SHARED_SIMULATION_APP = None
    yield
    isaac_session._SHARED_ISAAC_RESOURCES.clear()
    isaac_session._SHARED_SIMULATION_APP = None


def test_shared_isaac_resources_live_until_worker_cleanup():
    app = object()
    events = []
    resource = object()
    isaac_session.register_shared_simulation_app(app)
    isaac_session.register_shared_isaac_resource(
        "weight_sort_env", resource, close=lambda value: events.append(value)
    )
    assert isaac_session.shared_isaac_resource("weight_sort_env") is resource
    isaac_session.close_shared_isaac_resources()
    assert events == [resource]
    assert isaac_session.shared_isaac_resource("weight_sort_env") is None


def test_shared_isaac_resources_require_an_app_and_unique_name():
    with pytest.raises(RuntimeError, match="registered application"):
        isaac_session.register_shared_isaac_resource("env", object(), close=lambda _: None)
    app = object()
    isaac_session.register_shared_simulation_app(app)
    isaac_session.register_shared_isaac_resource("env", object(), close=lambda _: None)
    with pytest.raises(RuntimeError, match="already registered"):
        isaac_session.register_shared_isaac_resource("env", object(), close=lambda _: None)


def test_recycle_required_is_a_runtime_error():
    assert issubclass(isaac_session.PersistentWorkerRecycleRequired, RuntimeError)
