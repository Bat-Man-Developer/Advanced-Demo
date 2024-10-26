<?php
    // Store Information
    $productimagename = "confusion_matrix";
    $productimagename1 = "feature_importance";
    $productimagename2 = "performance_metrics";
    $productimagename3 = "roc_curve";
    $productimagename4 = "precision_recall_curve";
    $productimagename5 = "calibration_plot";
    $productimagename6 = "learning_curves";

    // Image Path
    $srcFilePath = $_SERVER['DOCUMENT_ROOT']."model_plots/";
    $specificFiles = array(
        $srcFilePath.$productimagename,
        $srcFilePath.$productimagename1,
        $srcFilePath.$productimagename2,
        $srcFilePath.$productimagename3,
        $srcFilePath.$productimagename4,
        $srcFilePath.$productimagename5,
        $srcFilePath.$productimagename6,
    );

    // Delete Image files in Directory
    foreach($specificFiles as $file){
        if(file_exists($file)){
            unlink($file);  
        }
    }