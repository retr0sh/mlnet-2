using Microsoft.ML;
using Microsoft.ML.Data;
using System.Reflection;


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


print_f(sdca_f(), "sdca");
print_f(fastree_f(), "fastree");
print_f(LbfgsPoissonRegression_f(), "LbfgsPoissonRegression");

ITransformer sdca_f() {
    var pipeline_sdca = mlContext.Transforms.Concatenate("Features", nameof(StudyData.StudyHours))
    .Append(mlContext.Regression.Trainers.Sdca(
        labelColumnName: "Label",
        maximumNumberOfIterations: 100));

    ITransformer model = pipeline_sdca.Fit(trainData);
    return model;
}

ITransformer fastree_f() {
    var pipeline_ft = mlContext.Transforms.Concatenate("Features", nameof(StudyData.StudyHours))
        .Append(mlContext.Regression.Trainers.FastTree(
            labelColumnName: "Label",
            numberOfLeaves: 6,
            numberOfTrees: 30,
            learningRate: 0.15f,
            minimumExampleCountPerLeaf: 1));

    ITransformer model = pipeline_ft.Fit(trainData);
    return model;
}

ITransformer LbfgsPoissonRegression_f() {
    var pipeline_ft = mlContext.Transforms.Concatenate("Features", nameof(StudyData.StudyHours))
        .Append(mlContext.Regression.Trainers.LbfgsPoissonRegression(
        labelColumnName: "Label"
        ));

    ITransformer model = pipeline_ft.Fit(trainData);
    return model;
}


void print_f(ITransformer model, string title) {
    Console.Write("--------------------  " + title + "  --------------------\n");
    IDataView predictions = model.Transform(testData);

    var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");

    Console.WriteLine($"Средняя абсолютная ошибка (MAE): {metrics.MeanAbsoluteError:F2}");
    Console.WriteLine($"R-квадрат (R^2): {metrics.RSquared:F2}");

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

    string end = "";
    for (int i = 0; i < title.Length+2; i++) { end += "-"; }
    Console.Write("---------------------" + end + "---------------------\n");
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
