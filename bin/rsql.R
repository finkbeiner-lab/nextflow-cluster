library(DBI)
library(RPostgres)

add_row <- function(tablename, dct, con) {
  # Create the SQL insert statement
  fields <- names(dct)
  values <- sapply(dct, function(val) {
    if (is.character(val)) {
      return(paste0("'", val, "'"))
    } else {
      return(as.character(val))
    }
  })
  sql <- paste0("INSERT INTO ", tablename, " (", paste(fields, collapse = ", "), ") VALUES (", paste(values, collapse = ", "), ")")
  
  # Execute the query
  dbExecute(con, sql)
  
  # In this version, we assume the caller manages the connection (including commits and rollbacks)
}

get_df <- function(tablename, sql_query) {
  con <- dbConnect(RPostgres::Postgres(), dbname = 'galaxy', host = 'fb-postgres01.gladstone.internal', port = 5432, user = 'postgres', password = pw)
  qry <- dbSendQuery(con, sql_query)
  df <- dbFetch(res)
  dbDisconnect(con)
  return df
}

get_row <- function(tablename, condition, con) {
  # Create the SQL select statement
  sql <- paste0("SELECT * FROM ", tablename, " WHERE ", condition, " LIMIT 1")
  
  # Execute the query and fetch the result
  result <- dbGetQuery(con, sql)
  
  return(result)
}

# Usage example
creds <- read.csv('/gladstone/finkbeiner/lab/GALAXY_INFO/pass.csv')
print(creds)
pw <- creds[1, "pw"]
print(pw)
con <- dbConnect(RPostgres::Postgres(), dbname = 'galaxy', host = 'fb-postgres01.gladstone.internal', port = 5432, user = 'postgres', password = pw)

# Example to get a row where "column1" is "value1"
row_data <- get_row("experimentdata", "experiment = '20231002-1-MSN-taueos'", con)
print(row_data)

dbDisconnect(con)

# # Usage example
# con <- dbConnect(RPostgres::Postgres(), dbname = 'galaxy', host = 'fb-postgres01.gladstone.internal:5432', port = 5432, user = 'postgres', password = 'your_password')
# data <- list(column1 = "value1", column2 = 123)
# add_row("experimentdata", data, con)
# dbDisconnect(con)