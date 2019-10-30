package com.packt.JavaDL.TitanicSurvival.Util;

import org.apache.spark.sql.SparkSession;

public final class SparkSessionUtil {
    private static volatile SparkSession INSTANCE;

    public static SparkSession getInstance() {
        if (null == INSTANCE) {
            synchronized (SparkSessionUtil.class) {
                if (null == INSTANCE)
                    INSTANCE = SparkSession
                            .builder()
                            .master("local[*]")
                            .config("spark.sql.warehouse.dir", "/home/mertins/temp/spark")
                            .appName("SurvivalPredictionMLP")
                            .getOrCreate();
            }
        }

        return INSTANCE;
    }
}
