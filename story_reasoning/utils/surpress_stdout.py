import contextlib
import io
import sys

@contextlib.contextmanager
def suppress_stdout():
    # Save the original stdout
    original_stdout = sys.stdout
    # Redirect stdout to a null device or string buffer
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        # Restore the original stdout
        sys.stdout = original_stdout