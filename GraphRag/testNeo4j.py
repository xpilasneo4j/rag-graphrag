import neo4j
import configparser

config = configparser.ConfigParser()
config.read('neo4j.env')

URI = config.get('Neo4j','neo4j_uri')
AUTH = ("neo4j", config.get('Neo4j','neo4j_password'))
DATABASE = config.get('Neo4j','neo4j_database')

driver = neo4j.GraphDatabase.driver(URI, auth=AUTH, database=DATABASE)

print("Success" if driver.verify_connectivity() == None else "Not connected")