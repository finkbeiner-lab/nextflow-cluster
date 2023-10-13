package com.finkbeiner;

// For convenience, always static import your generated tables and jOOQ functions to decrease verbosity:
import static org.jooq.impl.DSL.*;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.jooq.DSLContext;

import java.io.FileReader;
import java.io.IOException;


import java.sql.*;
import java.util.UUID;
import org.jooq.*;
import org.jooq.impl.*;

public class db {
    public static void main(String[] args) {
        String userName = "postgres";
        String password = "";

        try (
            FileReader reader = new FileReader("/gladstone/finkbeiner/lab/GALAXY_INFO/pass.csv");
            CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT.withHeader());
        ) {
            // Get the first record and the value from the "pw" column
            CSVRecord firstRecord = csvParser.getRecords().get(0);
            password = firstRecord.get("pw");                  
                
        } catch (IOException e) {
            e.printStackTrace();
        }

        String url = "jdbc:postgresql://fb-postgres01.gladstone.internal:5432/galaxy";

        // Connection is the only JDBC resource that we need
        // PreparedStatement and ResultSet are handled by jOOQ, internally
        try (Connection conn = DriverManager.getConnection(url, userName, password)) {
            DSLContext create = DSL.using(conn, SQLDialect.POSTGRES);
            Result<Record> result = create.select().from("experimentdata").fetch();

            for (Record r : result) {
                UUID uuid = r.get(field("id", UUID.class));
                String experiment = r.get(field("experiment", String.class));


                System.out.println("ID: " + uuid + " experiment: " + experiment);
            }
        } 
    
        catch (Exception e) {
            e.printStackTrace();
        }
    }
}