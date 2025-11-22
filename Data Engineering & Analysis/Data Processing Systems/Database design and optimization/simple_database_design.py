#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "pandas",
# ]
# ///

"""
Simple database design and optimization demonstration.
Demonstrates: schema design, indexing, queries, and performance optimization.
"""

import sqlite3
import pandas as pd
import time
import random

# Create in-memory database
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

print("Database Design and Optimization Demo")
print("=" * 70)

# Step 1: Create tables with proper schema
print("\n[1] Creating Database Schema")

cursor.execute("""
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

cursor.execute("""
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL,
    stock_quantity INTEGER DEFAULT 0
)
""")

cursor.execute("""
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
)
""")

print("   ✓ Created tables: users, products, orders")

# Step 2: Insert sample data
print("\n[2] Populating Database with Sample Data")

# Insert users
users_data = [
    (f'user_{i}', f'user{i}@example.com')
    for i in range(1000)
]
cursor.executemany("INSERT INTO users (username, email) VALUES (?, ?)", users_data)

# Insert products
categories = ['Electronics', 'Clothing', 'Books', 'Food', 'Toys']
products_data = [
    (f'Product {i}', random.choice(categories), random.uniform(10, 1000), random.randint(0, 100))
    for i in range(500)
]
cursor.executemany("INSERT INTO products (name, category, price, stock_quantity) VALUES (?, ?, ?, ?)", products_data)

# Insert orders
orders_data = [
    (random.randint(1, 1000), random.randint(1, 500), random.randint(1, 5))
    for _ in range(5000)
]
cursor.executemany("INSERT INTO orders (user_id, product_id, quantity) VALUES (?, ?, ?)", orders_data)

conn.commit()

print(f"   ✓ Inserted 1,000 users")
print(f"   ✓ Inserted 500 products")
print(f"   ✓ Inserted 5,000 orders")

# Step 3: Query performance WITHOUT indexes
print("\n[3] Query Performance Analysis")

query = """
SELECT u.username, p.name, o.quantity, o.order_date
FROM orders o
JOIN users u ON o.user_id = u.user_id
JOIN products p ON o.product_id = p.product_id
WHERE p.category = 'Electronics'
ORDER BY o.order_date DESC
LIMIT 100
"""

start = time.time()
cursor.execute(query)
results = cursor.fetchall()
time_without_index = time.time() - start

print(f"\n   Query WITHOUT indexes:")
print(f"   Time: {time_without_index*1000:.2f} ms")
print(f"   Results: {len(results)} rows")

# Step 4: Add indexes
print("\n[4] Creating Indexes for Optimization")

cursor.execute("CREATE INDEX idx_orders_user_id ON orders(user_id)")
cursor.execute("CREATE INDEX idx_orders_product_id ON orders(product_id)")
cursor.execute("CREATE INDEX idx_products_category ON products(category)")
cursor.execute("CREATE INDEX idx_orders_date ON orders(order_date)")

print("   ✓ Created indexes on foreign keys and frequently queried columns")

# Query performance WITH indexes
start = time.time()
cursor.execute(query)
results = cursor.fetchall()
time_with_index = time.time() - start

print(f"\n   Query WITH indexes:")
print(f"   Time: {time_with_index*1000:.2f} ms")
print(f"   Results: {len(results)} rows")
print(f"   Speedup: {time_without_index/time_with_index:.2f}x")

# Step 5: Advanced queries
print("\n[5] Advanced Query Examples")

# Aggregation query
cursor.execute("""
SELECT p.category, COUNT(*) as order_count, SUM(o.quantity * p.price) as total_revenue
FROM orders o
JOIN products p ON o.product_id = p.product_id
GROUP BY p.category
ORDER BY total_revenue DESC
""")

print("\n   Revenue by Category:")
for row in cursor.fetchall():
    print(f"     {row[0]:20s} | Orders: {row[1]:5d} | Revenue: ${row[2]:10,.2f}")

# Top customers query
cursor.execute("""
SELECT u.username, COUNT(*) as order_count, SUM(o.quantity) as total_items
FROM orders o
JOIN users u ON o.user_id = u.user_id
GROUP BY u.user_id
ORDER BY order_count DESC
LIMIT 5
""")

print("\n   Top 5 Customers:")
for i, row in enumerate(cursor.fetchall(), 1):
    print(f"     {i}. {row[0]:15s} | Orders: {row[1]:3d} | Items: {row[2]:4d}")

# Step 6: Query optimization tips
print("\n[6] Database Optimization Best Practices")
print("   ✓ Create indexes on foreign keys")
print("   ✓ Index frequently queried columns (WHERE, ORDER BY)")
print("   ✓ Use EXPLAIN QUERY PLAN to analyze query execution")
print("   ✓ Avoid SELECT *; specify only needed columns")
print("   ✓ Use JOINs instead of subqueries when possible")
print("   ✓ Normalize data to reduce redundancy")
print("   ✓ Use appropriate data types")

# Demonstrate EXPLAIN QUERY PLAN
print("\n[7] Query Execution Plan")
cursor.execute(f"EXPLAIN QUERY PLAN {query}")
print("\n   Execution plan:")
for row in cursor.fetchall():
    print(f"     {row}")

# Export to pandas for analysis
print("\n[8] Exporting to Pandas DataFrame")
df = pd.read_sql_query("""
    SELECT p.category, AVG(p.price) as avg_price, COUNT(*) as product_count
    FROM products p
    GROUP BY p.category
""", conn)

print("\n   Product Statistics by Category:")
print(df.to_string(index=False))

# Cleanup
conn.close()

print("\n" + "=" * 70)
print("Database demo completed!")
