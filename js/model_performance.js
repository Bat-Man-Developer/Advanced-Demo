class Performance {
    constructor() {
        this.message = null;
    }

    initializePlots() {
        const plots = [
            { id: 'confusion-matrix', filename: '../../model_plots/confusion_matrix.png' },
            { id: 'feature-importance', filename: '../../model_plots/feature_importance.png' },
            { id: 'performance-metrics', filename: '../../model_plots/performance_metrics.png' },
            { id: 'roc-curve', filename: '../../model_plots/roc_curve.png' },
            { id: 'precision-recall-curve', filename: '../../model_plots/precision_recall_curve.png' },
            { id: 'calibration-plot', filename: '../../model_plots/calibration_plot.png' },
            { id: 'learning-curves', filename: '../../model_plots/learning_curves.png' }
        ];
    
        plots.forEach(plot => {
            const img = document.getElementById(plot.id);
            if (img) {
                img.src = plot.filename;
                img.onerror = function() {
                    this.src = '../../model_plots/placeholder.png';
                    this.alt = 'No Image Found';
                };
            }
        });
    }

    getPerformance() {
        $.ajax({
            url: '../../call_python/getModelPerformance.php',
            method: 'GET',
            success: (response) => {
                const values = response.split(','); // Assuming ',' is the delimiter
                
                this.message = values[0];
                $('#responseMessage').html(this.message);

                this.initializePlots();
            },
            error: (xhr, status, error) => {
                console.error('Error getting Model Statistics:', error);
            }
        });
    }
}

$(document).ready(function() {
    const performance = new Performance();
    performance.getPerformance();
});