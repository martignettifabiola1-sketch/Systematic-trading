from subprocess import check_call
import sys
check_call([sys.executable, "src/h2_stub.py"])
print("OK: pesi generati in outputs/weights/")
