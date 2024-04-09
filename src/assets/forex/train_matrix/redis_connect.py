import redis
#from urllib.parse import urlparse  



REDIS_URL = "redis://default:zHeoOL4uqpzaxTC7YgtuWvq4HRNSsoD0@redis-17905.c326.us-east-1-3.ec2.cloud.redislabs.com:17905"

# Create a Redis connection
url_connection = redis.from_url(REDIS_URL)

# Test the connection
response = url_connection.ping()
print(response)
