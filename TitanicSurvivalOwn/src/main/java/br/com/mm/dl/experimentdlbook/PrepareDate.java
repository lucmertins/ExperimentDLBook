package br.com.mm.dl.experimentdlbook;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;

import java.io.Serializable;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.feature.StandardScalerModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;
import org.apache.spark.mllib.stat.Statistics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;

import scala.Option;
import scala.Some;
import scala.Tuple2;
import scala.Tuple3;

public class PrepareDate {

    public static UDF1<String, Option<Integer>> normEmbarked = (String d) -> {
        if (d == null) {
            return Option.apply(null);
        } else {
            if (d.equals("S")) {
                return Some.apply(0);
            } else if (d.equals("C")) {
                return Some.apply(1);
            } else {
                return Some.apply(2);
            }
        }
    };

    public static UDF1<String, Option<Integer>> normSex = (String d) -> {
        if (null == d) {
            return Option.apply(null);
        } else {
            if (d.equals("male")) {
                return Some.apply(0);
            } else {
                return Some.apply(1);
            }
        }
    };

    public void done() {
        SparkSession spark = SparkSession.builder().master("local[*]").config("spark.sql.warehouse.dir", "/home/mertins/temp/spark").appName("SurvivalPredictionMLP").getOrCreate();

        Dataset df = spark.sqlContext()
                .read()
                .format("com.databricks.spark.csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("/home/mertins/Desenvolvimento/Java/DeepLearning/ExperimentDLBook/TitanicSurvival/data/train.csv");

        spark.sqlContext().udf().register("normEmbarked", normEmbarked, DataTypes.IntegerType);
        spark.sqlContext().udf().register("normSex", normSex, DataTypes.IntegerType);

        Dataset projection = df.select(
                col("Survived"),
                col("Fare"),
                callUDF("normSex", col("Sex")).alias("Sex"),
                col("Age"),
                col("Pclass"),
                col("Parch"),
                col("SibSp"),
                callUDF("normEmbarked", col("Embarked")).alias("Embarked")
        );

        JavaRDD<Vector> statsDf = projection.rdd().toJavaRDD().map(row -> {
            Row r = (Row) row;
            return Vectors.dense(
                    r.<Double>getAs("Fare"),
                    r.isNullAt(3) ? 0d : r.<Double>getAs("Age"));
        });

        MultivariateStatisticalSummary summary = Statistics.colStats(statsDf.rdd());

        double meanFare = summary.mean().apply(0);
        double meanAge = summary.mean().apply(1);

        
        Vector stddev=Vectors.dense(Math.sqrt(summary.variance().apply(0)),Math.sqrt(summary.variance().apply(1)));
        Vector mean=Vectors.dense(summary.mean().apply(0),summary.mean().apply(1));
        StandardScalerModel scaler=new StandardScalerModel(stddev, mean);
        
        
        

        projection.show(50);
        System.out.printf("%f %f %f %f\n", meanFare, meanAge,summary.variance().apply(0),summary.variance().apply(1));
    }
    public static void main(String[] args) {
        PrepareDate pd=new PrepareDate();
        pd.done();
    }
}
