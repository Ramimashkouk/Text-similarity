from TextSimModule import text_similarity
from time import time

start_time = time()

with open ('file1.txt') as f:
    f1 = f.read()
with open ('file2.txt') as f:
    f2 = f.read()
documents = (f1, f2)
text_similarity(documents)

passed_time = (time() - start_time)/60
print()
print(f"It took {passed_time} Minutes")
