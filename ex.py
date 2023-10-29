import re

# Read the SQL file containing INSERT statements
with open('/Users/jenny/Downloads/food-database-0-1000.sql', 'r') as file:
    sql_content = file.read()

# Using regular expressions to extract the product name
pattern = r"values\('([^']+)'"
product_name = re.search(pattern, sql_insert_statement).group(1)

# Handling potential string issues
escaped_product_name = product_name.replace("'", "''")

# Generating a new SQL INSERT statement with the escaped string
new_sql_insert_statement = f"insert into product (product_name) values('{escaped_product_name}')"

print(new_sql_insert_statement)

