import jax
import pytest

setup_done = False

def initialize_distributed():
  global setup_done
  if not setup_done:
    try:
      jax.distributed.initialize()
    finally:
      print(f"JAX distributed is not activated, testing only on single device")

    print(f"number of devices: {jax.device_count()}")
    setup_done = True


@pytest.fixture(scope="session", autouse=True)
def setup_and_teardown_session():
  # Code to run at the start of the session
  print("Starting session...")
  initialize_distributed()
  # Setup code here
  # e.g., connecting to a database, initializing some resources, etc.

  yield

  # Code to run at the end of the session
  print("Ending session...")
  jax.distributed.shutdown()

  # Teardown code here
  # e.g., closing connections, cleaning up resources, etc.

 
