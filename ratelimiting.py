import time

requests = []
LIMIT = 3   
WINDOW = 10  

def is_allowed():
    current_time = time.time()

    # remove old requests
    while requests and current_time - requests[0] > WINDOW:
        requests.pop(0)

    if len(requests) < LIMIT:
        requests.append(current_time)
        return True
    else:
        return False

for i in range(5):
    if is_allowed():
        print("Request allowed")
    else:
        print("Rate limit exceeded")
    time.sleep(2)