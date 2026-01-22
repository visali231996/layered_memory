import time
import redis

HOST = "localhost"
PORT = 6379

KEY = "ttl:test_key"
VALUE = "hello"
TTL_SECONDS = 10  # change to whatever you want

r = redis.Redis(host=HOST, port=PORT, decode_responses=True)

# 1) sanity check
print("PING:", r.ping())

# 2) set key with TTL
r.set(KEY, VALUE, ex=TTL_SECONDS)
print(f"SET {KEY}='{VALUE}' with TTL={TTL_SECONDS}s")

# 3) show TTL immediately
print("TTL right after set:", r.ttl(KEY), "seconds")
print("GET right after set:", r.get(KEY))

# 4) wait for expiry (+2s buffer)
wait = TTL_SECONDS + 2
print(f"Sleeping {wait}s...")
time.sleep(wait)

# 5) check again
print("TTL after sleep:", r.ttl(KEY), "seconds")  # -2 means key doesn't exist
print("GET after sleep:", r.get(KEY))             # should be None
print("EXISTS after sleep:", r.exists(KEY))       # should be 0