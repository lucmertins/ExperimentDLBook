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
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.DataFrameReader;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;

import scala.Option;
import scala.Some;
import scala.Tuple2;
import scala.Tuple3;

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

    public static Tuple3<Double, Double, Double> flattenPclass(double value) {
        Tuple3<Double, Double, Double> result;

        if (value == 1) {
            result = new Tuple3<>(1d, 0d, 0d);
        } else if (value == 2) {
            result = new Tuple3<>(0d, 1d, 0d);
        } else {
            result = new Tuple3<>(0d, 0d, 1d);
        }

        return result;
    }

    public static Tuple3<Double, Double, Double> flattenEmbarked(double value) {
        Tuple3<Double, Double, Double> result;

        if (value == 0) {
            result = new Tuple3<>(1d, 0d, 0d);
        } else if (value == 1) {
            result = new Tuple3<>(0d, 1d, 0d);
        } else {
            result = new Tuple3<>(0d, 0d, 1d);
        }

        return result;
    }

    public static Tuple2<Double, Double> flattenSex(double value) {
        Tuple2<Double, Double> result;

        if (value == 0) {
            result = new Tuple2<>(1d, 0d);
        } else {
            result = new Tuple2<>(0d, 1d);
        }

        return result;
    }

    public void done() {
        SparkSession spark = SparkSession.builder().master("local[*]").config("spark.sql.warehouse.dir", "/home/mertins/temp/spark").appName("SurvivalPredictionMLP").getOrCreate();
        DataFrameReader dataFrame = spark.sqlContext()
                .read()
                .format("com.databricks.spark.csv")
                .option("header", "true")
                .option("inferSchema", "true");
        
        Dataset dataSet = dataFrame.load("/home/mertins/Desenvolvimento/Java/DeepLearning/ExperimentDLBook/TitanicSurvival/data/train.csv");

        dataSet.filter(col("Embarked").isNull()).show();
        
        Dataset<Row> projection1 = dataSet.select(
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
                return Option.apply(0);
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

        Dataset<Row> projection2 = dataSet.select(
                col("Survived"),
                callUDF("normFare", col("Fare").cast("string")).alias("Fare"),
                callUDF("normSex", col("Sex")).alias("Sex"),
                callUDF("normAge", col("Age").cast("string")).alias("Age"),
                col("Pclass"),
                col("Parch"),
                col("SibSp"),
                callUDF("normEmbarked", col("Embarked")).alias("Embarked")
        );

        Vector stddev = Vectors.dense(Math.sqrt(summary.variance().apply(0)), Math.sqrt(summary.variance().apply(1)));
        Vector mean = Vectors.dense(summary.mean().apply(0), summary.mean().apply(1));
        StandardScalerModel scaler = new StandardScalerModel(stddev, mean);

        Encoder<Integer> integerEncoder = Encoders.INT();
        Encoder<Double> doubleEncoder = Encoders.DOUBLE();

        Encoder<Vector> vectorEncoder = Encoders.kryo(Vector.class);
        Encoders.tuple(integerEncoder, vectorEncoder);
        Encoders.tuple(doubleEncoder, vectorEncoder);

//        projection2.show(10000);

        JavaRDD<VectorPair> scaledRDD = projection2.toJavaRDD().map(row -> {
            VectorPair vectorPair = new VectorPair();
            org.apache.spark.mllib.linalg.Vector scaledContinous = scaler.transform(Vectors.dense(row.<Double>getAs("Fare"), row.<Double>getAs("Age")));
            Tuple3<Double, Double, Double> pclassFlat = flattenPclass(row.<Integer>getAs("Pclass"));
            Tuple3<Double, Double, Double> embarkedFlat = flattenEmbarked(row.<Integer>getAs("Embarked"));
            Tuple2<Double, Double> sexFlat = flattenSex(row.<Integer>getAs("Sex"));

            vectorPair.setLable(new Double(row.<Integer>getAs("Survived")));
            Vector dense = Vectors.dense(
                    scaledContinous.apply(0),
                    scaledContinous.apply(1),
                    sexFlat._1(),
                    sexFlat._2(),
                    pclassFlat._1(),
                    pclassFlat._2(),
                    pclassFlat._3(),
                    embarkedFlat._1(),
                    embarkedFlat._2(),
                    embarkedFlat._3());

            vectorPair.setFeatures(dense);
            return vectorPair;
        });

        Dataset<Row> scaleDF = spark.createDataFrame(scaledRDD, VectorPair.class);

        Dataset<Row> scaleDF2 = MLUtils.convertVectorColumnsToML(scaleDF);

        Dataset<Row> data = scaleDF2.toDF("features", "label");

        Dataset<Row>[] datasets = data.randomSplit(new double[]{.8, .2}, 12345L);
//
        Dataset<Row> training = datasets[0];
        Dataset<Row> validation = datasets[1];

        projection1.show(50);
        projection2.show(50);
        System.out.printf("%f %f %f %f\n", meanFare, meanAge, summary.variance().apply(0), summary.variance().apply(1));
        scaleDF.show(50);
        scaleDF2.show(50);
        training.show();
    }

    public static void main(String[] args) {
        PrepareDate pd = new PrepareDate();
        pd.done();
    }
}
