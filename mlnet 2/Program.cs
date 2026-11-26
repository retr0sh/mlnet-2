using Microsoft.ML;
using Microsoft.ML.Data;


var mlContext = new MLContext();

string dataPath = "data.csv";
IDataView dataView = mlContext.Data.LoadFromTextFile<StudyData>(
    path: dataPath,
    hasHeader: true,
    separatorChar: ','
);

var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
IDataView trainData = split.TrainSet;
IDataView testData = split.TestSet;

var pipeline = mlContext.Transforms.Concatenate("Features", nameof(StudyData.StudyHours))
    .Append(mlContext.Regression.Trainers.FastTree(
        labelColumnName: "Label",
        numberOfLeaves: 6,
        numberOfTrees: 30,
        learningRate: 0.15f,
        minimumExampleCountPerLeaf: 1));

ITransformer model = pipeline.Fit(trainData);


IDataView predictions = model.Transform(testData);

var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");

Console.WriteLine($"Средняя абсолютная ошибка (MAE): {metrics.MeanAbsoluteError:F2}");
Console.WriteLine($"R-квадрат (R²): {metrics.RSquared:F2}");

var predictionEngine = mlContext.Model.CreatePredictionEngine<StudyData, ScorePrediction>(model);

StudyData[] list_sh = {
    new StudyData { StudyHours = 3f },
    new StudyData { StudyHours = 5.5f },
    new StudyData { StudyHours = 6f },
    new StudyData { StudyHours = 10f }
};


for (int i = 0; i < 4; i++) {
    Console.WriteLine("");

    var input = list_sh[i];
    var prediction = predictionEngine.Predict(input);

    Console.WriteLine($"Если готовиться {input.StudyHours} часов, прогноз оценки: {prediction.PredictedScore:F1}");
}

public class StudyData {
    [LoadColumn(0)]
    public float StudyHours { get; set; }

    [LoadColumn(1)]
    [ColumnName("Label")]
    public float Score { get; set; }
}

public class ScorePrediction{
    [ColumnName("Score")]
    public float PredictedScore { get; set; }
}