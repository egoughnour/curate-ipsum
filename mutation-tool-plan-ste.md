# Goal

Build a small HTTP JSON service called "mutation-tool-server".
The service runs unit tests and mutation tests.
The service records run results in a local store.
The service exposes history and region metrics endpoints.
Another agent will call these endpoints through MCP.

## High level design

Use Node.js and TypeScript.
Use Express for the HTTP API.
Use child processes to run external tools.
Make all behavior driven by configuration and request payloads.
Do not hard-code paths.

## Services and modules

Create these modules:

1. src/types.ts

Define TypeScript interfaces for:
  - RunMeta
  - TestRunResult
  - FileMutationStats
  - MutationRunResult
  - PIDComponents
  - RegionMetrics
  - RunHistory

2. src/processRunner.ts

Implement a function runCommand(options) that runs a shell command.
The function uses child_process.spawn.
The function captures stdout and stderr.
The function measures duration in milliseconds.
The function returns:
- exit code
- stdout
- stderr
- durationMs
Handle command errors.
Do not crash the process.

3. src/historyStore.ts.  
Implement a local history store.
Use a directory from environment variable MUTATION_TOOL_DATA_DIR.
If the variable is not set, use .mutation_tool_data under the current working directory.
Store runs in a JSONL file runs.jsonl.
Provide these functions:
- appendRun(run)
- getRunHistory(projectId, regionId, limit)
  appendRun adds one line with JSON to the file.
  getRunHistory reads the file, parses runs, filters, sorts by timestamp, and returns the newest limit runs.

4. src/strykerParser.ts

Implement logic to parse Stryker JSON output.
Accept raw stdout and stderr from the runCommand call.
Accept the working directory path.
Find the Stryker report file under the working directory.
The location and name should be configurable if possible.
Parse the JSON report into MutationRunResult shape.
Populate:
- totalMutants
- killed
- survived
- noCoverage
- mutationScore (0..1)
- byFile with per-file stats
If the report is missing or invalid, return a clear error.

5. src/testOutputParser.ts

Implement logic to parse unit test output.
Support at least one framework, for example dotnet test.
Accept stdout, stderr, and exit code.
Extract:
- total tests
- passed tests
- failed tests
- failing test names if possible
Map this data into a TestRunResult.

6. src/regionMetrics.ts

Implement a module that computes RegionMetrics.
For now, use a simple implementation.
Take as input:
- projectId
- commitSha
- regionId
- run history for this project and region
Compute mutationScore as the latest mutation score for this region.
Compute centrality as a placeholder constant (for example 0.5).
Compute triviality as a placeholder constant (for example 0.5).
Compute PID components:
- P is (1 - current mutationScore)
- I is the sum of (1 - score) over the last N runs for this region with exponential decay
- D is difference between the latest two (1 - score) values
Make N configurable.
Return a RegionMetrics object.

7. src/server.ts

Implement an Express server.
Use express.json() for JSON bodies.
Expose these POST endpoints:
- /run-unit-tests
- /run-integration-tests
- /run-mutation-tests
- /get-run-history
- /get-region-metrics

**/run-unit-tests**

Request body:
- projectId
- commitSha
- command
- workingDirectory
- regionId (optional)
Use runCommand with the given command and working directory.
Use testOutputParser to build a TestRunResult.
Set kind to "unit".
Set id to a new UUID.
Set timestamp to current time in ISO format.
Append the run to history.
Return the run as JSON.

**/run-integration-tests**

Same contract as /run-unit-tests.
Only change kind to "integration".

**/run-mutation-tests**

Request body:
- projectId
- commitSha
- command
- workingDirectory
- regionId (optional)
Use runCommand to run the Stryker command.
Use strykerParser to convert output to a MutationRunResult.
Set id and timestamp.
Append the run to history.
Return the run as JSON.

**/get-run-history**

Request body:
- projectId
- regionId (optional)
- limit (optional)
Use getRunHistory from the history store.
Return the result as JSON.

**/get-region-metrics**

Request body:
- projectId
- commitSha
- regionId
Load run history for this project and region.
Pass the data to regionMetrics module.
Return the RegionMetrics object as JSON.

For all endpoints:
- Validate request body and handle missing fields.
Log errors with enough detail for debugging.
Return HTTP 400 for invalid requests.
Return HTTP 500 for internal errors.

8. package.json and tsconfig.json

Create a package.json that declares dependencies and scripts.
Include scripts:
- "build" for TypeScript compilation
- "start" to run node dist/server.js
- "dev" to run ts-node src/server.ts
Configure tsconfig.json to compile to CommonJS or ES modules as needed.

### Testing

Add a simple test or script that:
- Starts the server.
- Calls /run-unit-tests with a trivial test command that always passes.
- Calls /run-mutation-tests with a dry-run or mocked Stryker config if needed.
- Calls /get-run-history.
- Calls /get-region-metrics.
Confirm that JSON responses match the TypeScript interfaces.

### Non-functional requirements

- Use clear logging.
- Avoid TODO comments.
- Avoid empty interfaces.
- Fail fast on configuration errors.
- Keep each module small and cohesive.