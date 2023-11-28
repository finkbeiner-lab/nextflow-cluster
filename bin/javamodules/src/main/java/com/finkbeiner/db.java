package com.finkbeiner;

// For convenience, always static import your generated tables and jOOQ functions to decrease verbosity:
import static org.jooq.impl.DSL.*;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.jooq.DSLContext;
import org.jooq.Condition;
import org.jooq.tools.jdbc.JDBCUtils;

import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import org.jooq.Record1;
import org.jooq.Table;

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
        Map<String, Object> conditions = new HashMap<>();
        Map<String, Object> updateMap = new HashMap<>();
        conditions.put("experiment", "testset");
        updateMap.put("centroid_x", 1000);
        // Connection is the only JDBC resource that we need
        // PreparedStatement and ResultSet are handled by jOOQ, internally
        getRows("experimentdata",conditions);
        Object exp_uuid = getTableUUID("experimentdata", conditions);
        
        Map<String, Object> conditions_well = new HashMap<>();
        conditions_well.put("experimentdata_id", exp_uuid);
        conditions_well.put("well", "A1");

        Object well_uuid = getTableUUID("welldata", conditions_well);


        Map<String, Object> conditions2 = new HashMap<>();
        conditions2.put("experimentdata_id", exp_uuid);
        conditions2.put("randomcellid", 1);
        conditions2.put("welldata_id", well_uuid);
        getRows("celldata",conditions2);

        update("celldata",updateMap, conditions2);
    }

    private static void getRows(String tablename, Map<String, Object> conditions) {
        try (Connection conn = DriverManager.getConnection(JDBC_URL, USERNAME, PASSWORD)) {
            DSLContext create = DSL.using(conn, SQLDialect.POSTGRES);
            System.out.println(conditions);
            Result<Record> result = create
                    .select()
                    .from(tablename)
                    .where(buildConditions(conditions))
                    //         ^^^^^^^^^^^^  ^^^^^^^^^^^ <-> ^^^^^  ^^^^^ Types must match
                    .fetch();

            for (Record r : result) {
                // UUID uuid = r.get(field("id", UUID.class));
                // String experiment = r.get(field("experiment", String.class));


                // System.out.println("ID: " + uuid + " experiment: " + experiment);
                System.out.println(r.toString());
            }
        } 
    
        catch (Exception e) {
            e.printStackTrace();
        }
    }


    public static Object getTableUUID(String tablename, Map<String, Object> conditions) {
        try (Connection conn = DriverManager.getConnection(JDBC_URL, USERNAME, PASSWORD)) {
            DSLContext create = DSL.using(conn, SQLDialect.POSTGRES);
            System.out.println(conditions);
            Result<?> result = create
            .select()
            .from(table(tablename))
            .where(buildConditions(conditions))
            .fetch();

        if (result.size() == 0) {
            return null;
        }
        Object uuid = result.get(0).get(field("id"));
        System.out.println(uuid);
        return uuid;
    }
        catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public static void update(String tablename, Map<String, Object> updateMap, Map<String, Object> conditions) {
        try (Connection conn = DriverManager.getConnection(JDBC_URL, USERNAME, PASSWORD)) {
            DSLContext create = DSL.using(conn, SQLDialect.POSTGRES);
            System.out.println(conditions);

            create
            .update(table(tablename))
            .set(updateMap)
            .where(buildConditions(conditions))
            .execute();
        }

        catch (Exception e) {
            e.printStackTrace();
        }
    }
    

    private static Condition buildConditions(Map<String, Object> conditions) {
        Condition condition = DSL.noCondition();
        for (Map.Entry<String, Object> entry : conditions.entrySet()) {
            condition = condition.and(field(entry.getKey()).eq(entry.getValue()));
        }
        return condition;
    }

}