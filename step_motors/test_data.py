#!/usr/bin/env python3

import subprocess
result = subprocess.run(['ls', '-l'], stdout=subprocess.PIPE)
print(result.stdout)
