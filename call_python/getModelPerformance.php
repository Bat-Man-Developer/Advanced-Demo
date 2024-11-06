<?php
set_time_limit(0);
ignore_user_abort(true);

class Model
{
    private $pythonExePath = "c:/Users/user/AppData/Local/Programs/Python/Python312/python.exe";
    private $scriptPath = "../python/model_performance.py";

    private function executeCommand()
    {
        $output = null;
        
        // Keep executing until we get a valid response
        while($output === null) {
            $escapedPythonScript = escapeshellarg($this->scriptPath);
            $fullCommand = sprintf('%s %s', 
                escapeshellarg($this->pythonExePath),
                $escapedPythonScript
            );
            
            $output = shell_exec($fullCommand);

            if ($output === null) {
                sleep(1); // Wait before retrying
            }
        }

        return trim($output);
    }

    public function getModel()
    {
        return $this->executeCommand();
    }
}

$model = new Model();
echo $model->getModel();