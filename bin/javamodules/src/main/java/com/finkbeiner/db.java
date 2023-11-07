package com.finkbeiner;

// For convenience, always static import your generated tables and jOOQ functions to decrease verbosity:
import static org.jooq.impl.DSL.*;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.jooq.DSLContext;
import org.jooq.tools.jdbc.JDBCUtils;

import java.io.FileReader;
import java.io.IOException;


import java.sql.*;
import java.util.Map;
import java.util.UUID;
import org.jooq.*;
import org.jooq.impl.*;

public class db {
    private static final String JDBC_URL = "jdbc:postgresql://fb-postgres01.gladstone.internal:5432/galaxy";
    private static final String USERNAME = "postgres";
    private static String PASSWORD;
    
    public static void main(String[] args) {


        try (
            FileReader reader = new FileReader("/gladstone/finkbeiner/lab/GALAXY_INFO/pass.csv");
            CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT.withHeader());
        ) {
            // Get the first record and the value from the "pw" column
            CSVRecord firstRecord = csvParser.getRecords().get(0);
            PASSWORD = firstRecord.get("pw");                  
                
        } catch (IOException e) {
            e.printStackTrace();
        }


        // Connection is the only JDBC resource that we need
        // PreparedStatement and ResultSet are handled by jOOQ, internally
        getRow("experimentdata","experiment","20231002-1-MSN-taueos");
    }

    private static void getRow(String tablename, String columnname, String value) {
        try (Connection conn = DriverManager.getConnection(JDBC_URL, USERNAME, PASSWORD)) {
            DSLContext create = DSL.using(conn, SQLDialect.POSTGRES);
            System.out.println(value);
            Result<Record> result = create
                    .select()
                    .from(tablename)
                    .where(field(columnname).eq(value))
                    //         ^^^^^^^^^^^^  ^^^^^^^^^^^ <-> ^^^^^  ^^^^^ Types must match
                    .fetch();

            String r = result.toString();
            System.out.println(r);

            // for (Record r : result) {
            //     // UUID uuid = r.get(field("id", UUID.class));
            //     // String experiment = r.get(field("experiment", String.class));


            //     // System.out.println("ID: " + uuid + " experiment: " + experiment);
            //     System.out.println(r.toString());
            // }
        } 
    
        catch (Exception e) {
            e.printStackTrace();
        }
    }


    // private static void addRow(DSLContext dslContext, String tableName, Map<String, String> data) {
    //     Table<?> table = table(DSL.name(tableName));

    //     for (Map.Entry<String, Object> entry : dct.entrySet()) {
    //         dslContext.insertInto(table).set(DSL.field(DSL.name(entry.getKey())), entry.getValue());
    //     }
    //     dslContext
    //         .insertInto(table(tableName))
    //         .set(field(data.keySet()), data.values())
    //         .execute();
    // }
}