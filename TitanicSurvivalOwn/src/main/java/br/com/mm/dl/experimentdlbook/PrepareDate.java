package br.com.mm.dl.experimentdlbook;

import java.io.Serializable;
import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.feature.StandardScalerModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;
import org.apache.spark.mllib.stat.Statistics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;

import scala.Option;
import scala.Some;

public class PrepareDate {

    public static class VectorPair implements Serializable {

        private double label;
        private Vector features;

        public VectorPair(double label, Vector features) {
            this.label = label;
            this.features = features;
        }

        public VectorPair() {
        }

        public void setFeatures(Vector features) {
            this.features = features;
        }

        public Vector getFeatures() {
            return this.features;
        }

        public void setLable(double label) {
            this.label = label;
        }

        public double getLable() {
            return this.label;
        }
    }

    public void done() {
        SparkSession spark = SparkSession.builder().master("local[*]").config("spark.sql.warehouse.dir", "/home/mertins/temp/spark").appName("SurvivalPredictionMLP").getOrCreate();

        Dataset df = spark.sqlContext()
                .read()
                .format("com.databricks.spark.csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load("/home/mertins/Desenvolvimento/Java/DeepLearning/ExperimentDLBook/TitanicSurvival/data/train.csv");

        Dataset<Row> projection1 = df.select(
                col("Survived"),
                col("Fare"),
                col("Sex"),
                col("Age"),
                col("Pclass"),
                col("Parch"),
                col("SibSp"),
                col("Embarked")
        );

        JavaRDD<Vector> statsDf = projection1.rdd().toJavaRDD().map(row -> {
            Row r = (Row) row;
            return Vectors.dense(
                    r.<Double>getAs("Fare"),
                    r.isNullAt(3) ? 0d : r.<Double>getAs("Age"));
        });

        MultivariateStatisticalSummary summary = Statistics.colStats(statsDf.rdd());
        double meanFare = summary.mean().apply(0);
        double meanAge = summary.mean().apply(1);

        UDF1<String, Option<Integer>> normEmbarked = (String d) -> {
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

        UDF1<String, Option<Integer>> normSex = (String d) -> {
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

        UDF1<String, Option<Double>> normFare = (String d) -> {
            if (null == d) {
                return Some.apply(meanFare);
            } else {
                return Some.apply(Double.parseDouble(d));
            }
        };

        UDF1<String, Option<Double>> normAge = (String d) -> {
            if (null == d) {
                return Some.apply(meanAge);
            } else {
                return Some.apply(Double.parseDouble(d));
            }
        };

        spark.sqlContext().udf().register("normEmbarked", normEmbarked, DataTypes.IntegerType);
        spark.sqlContext().udf().register("normSex", normSex, DataTypes.IntegerType);
        spark.sqlContext().udf().register("normFare", normFare, DataTypes.DoubleType);
        spark.sqlContext().udf().register("normAge", normAge, DataTypes.DoubleType);

        Dataset<Row> projection2 = df.select(
                col("Survived"),
                callUDF("normFare", col("Fare").cast("string")).alias("Fare"),
                callUDF("normSex", col("Sex")).alias("Sex"),
                callUDF("normAge", col("Age").cast("string")).alias("Age"),
                col("Pclass"),
                col("Parch"),
                col("SibSp"),
                callUDF("normEmbarked", col("Embarked")).alias("Embarked")
        );

//        Vector stddev = Vectors.dense(Math.sqrt(summary.variance().apply(0)), Math.sqrt(summary.variance().apply(1)));
//        Vector mean = Vectors.dense(summary.mean().apply(0), summary.mean().apply(1));
//        StandardScalerModel scaler = new StandardScalerModel(stddev, mean);
//
//        Encoder<Integer> integerEncoder = Encoders.INT();
//        Encoder<Double> doubleEncoder = Encoders.DOUBLE();
//
//        Encoder<Vector> vectorEncoder = Encoders.kryo(Vector.class);
//        Encoders.tuple(integerEncoder, vectorEncoder);
//        Encoders.tuple(doubleEncoder, vectorEncoder);
//
//        Dataset<Row> finalDF = projection.select(
//                col("Survived"),
//                callUDF("normFare", col("Fare").cast("string")).alias("Fare"),
//                col("Sex"),
//                callUDF("normAge", col("Age").cast("string")).alias("Age"),
//                col("Pclass"),
//                col("Parch"),
//                col("SibSp"),
//                col("Embarked"));
//
//        JavaRDD<VectorPair> scaledRDD = projection.toJavaRDD().map(row -> {
//            VectorPair vectorPair = new VectorPair();
//            vectorPair.setLable(new Double(row.<Integer>getAs("Survived")));
//            vectorPair.setFeatures(Util.getScaledVector(
//                    row.<Double>getAs("Fare"),
//                    row.<Double>getAs("Age"),
//                    row.<Integer>getAs("Pclass"),
//                    row.<Integer>getAs("Sex"),
//                    row.isNullAt(7) ? 0d : row.<Integer>getAs("Embarked"),
//                    scaler));
//
//            return vectorPair;
//        });

        projection1.show(50);
        projection2.show(50);
        System.out.printf("%f %f %f %f\n", meanFare, meanAge, summary.variance().apply(0), summary.variance().apply(1));
    }

    public static void main(String[] args) {
        PrepareDate pd = new PrepareDate();
        pd.done();
    }
}
