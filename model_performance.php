<?php
  include('layouts/header.php');
  include('server/delete_model_plots.php');
?>
  <body>
    <!--------- Model Performance-Page ------------>
    <section class="dashboard">
      <div class="dashboardcontainer" id="dashboardcontainer">
        <div class="text-center mt-3 pt-5 col-lg-6 col-md-12 col-sm-12">
          <p class="text-center" style="color: green"><?php if(isset($_GET['registermessage'])){ echo $_GET['registermessage']; }?></p>
          <p class="text-center" style="color: green"><?php if(isset($_GET['loginmessage'])){ echo $_GET['loginmessage']; }?></p>
          <p class="text-center" style="color: red"><?php if(isset($_GET['errormessage'])){ echo $_GET['errormessage']; }?></p>
          <h3>MACHINE LEARNING MODEL PERFORMANCE</h3>
          <hr class="mx-auto">
          <div class="dashboardadmininfo" id="dashboardadmininfo">
          </div>
          <div class="dashboardinfo" id="dashboardinfo">
            <div class="admindashboardcontainer">
              <div class="adminidscontainer">
                <div class="adminidstable">
                  <h1 class="adminidstitle"></h1>
                  <h2 class="adminidstitle"><p id="localnetworktraffictime"></p></h2>
                  <h3 style="color: blue; font-size: small"><p id="responseMessage">Loading Model Performance...</p></h3>
                  <div class="adminIdsBtnNav">
                    <a class="btn" href="index.php">Home</a>
                    <a class="btn" href="model_performance.php">Model Performance</a>
                    <a class="btn" href="model_performance.php">Refresh</a>
                  </div>
                  <main>
                    <h1>Model Performance</h1>
                    <div class="container">
                      <div class="plot-container">
                        <div class="plot">
                            <h2>Confusion Matrix</h2>
                            <img id="confusion-matrix" alt="Loading Image...">
                        </div>
                        <div class="plot">
                            <h2>Model Weights</h2>
                            <img id="model-weights" alt="Loading Image...">
                        </div>
                        <div class="plot">
                            <h2>Feature Importance</h2>
                            <img id="feature-importance" alt="Loading Image...">
                        </div>
                        <div class="plot">
                            <h2>Performance Metrics</h2>
                            <img id="performance-metrics" alt="Loading Image...">
                        </div>
                        <div class="plot">
                            <h2>ROC Curve</h2>
                            <img id="roc-curve" alt="Loading Image...">
                        </div>
                        <div class="plot">
                            <h2>Precision-Recall Curve</h2>
                            <img id="precision-recall-curve" alt="Loading Image...">
                        </div>
                        <div class="plot">
                            <h2>Calibration Plot</h2>
                            <img id="calibration-plot" alt="Loading Image...">
                        </div>
                        <div class="plot">
                            <h2>Boxplots Predictions</h2>
                            <img id="boxplots-predictions" alt="Loading Image...">
                        </div>
                        <div class="plot">
                            <h2>Accuracy Comparison</h2>
                            <img id="accuracy-comparison" alt="Loading Image...">
                        </div>
                        <div class="plot">
                            <h2>Precision Comparison</h2>
                            <img id="precision-comparison" alt="Loading Image...">
                        </div>
                        <div class="plot">
                            <h2>Recall Comparison</h2>
                            <img id="recall-comparison" alt="Loading Image...">
                        </div>
                        <div class="plot">
                            <h2>F1-Score Comparison</h2>
                            <img id="f1score-comparison" alt="Loading Image...">
                        </div>
                        <div class="plot">
                            <h2>Bias - Variance Tradeoff</h2>
                            <img id="bias-variance-tradeoff" alt="Loading Image...">
                        </div>
                        <div class="plot">
                            <h2>Learning Curves</h2>
                            <img id="learning-curves" alt="Loading Image...">
                        </div>
                      </div>
                    </div>
                  </main>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section><br><br><br><br><br><br><br><br><br>
    <script src="js/model_performance.js"></script>
  </body>
</html>