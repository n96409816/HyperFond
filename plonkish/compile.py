#!/usr/python
import subprocess

com = '''
cd benchmark/
cargo build --example basefold_proof_system --release
'''
print(com)
subprocess.call(["bash", "-c", com])


com = '''
cd benchmark/
cargo build --example distributed_basefold_proof_system --release
'''
print(com)
subprocess.call(["bash", "-c", com])