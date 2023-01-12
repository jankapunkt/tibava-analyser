# import uuid

# from datetime import datetime

# from analyser.plugin.databases.redis_database import RedisDatabase
# from analyser.data import generate_id, create_data_path


# def test_check_data():

#     database = RedisDatabase()
#     data_id = generate_id()
#     original_data = {"last_access": datetime.now().isoformat(), "data_id": data_id}
#     database.set(data_id, original_data)

#     data = database.get(data_id)
#     assert data.get("data_id") == original_data.get("data_id")
#     assert data.get("last_access") == original_data.get("last_access")

#     assert data_id in database.keys()

#     founded = False
#     for k, v in database:
#         if k == data_id:
#             assert v.get("data_id") == original_data.get("data_id")
#             assert v.get("last_access") == original_data.get("last_access")
#             founded = True

#     assert founded

#     assert not database.delete(generate_id())
#     assert database.delete(data_id)
