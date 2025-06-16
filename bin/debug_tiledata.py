from sql import Database

db = Database()

print("\n🔍 Columns in 'tiledata' table:")
print(list(db.meta.tables['tiledata'].columns.keys()))
