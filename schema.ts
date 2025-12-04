export type RunKind = 'unit' | 'integration' | 'mutation';

export interface RunMeta {
  id: string;
  projectId: string;
  commitSha: string;
  regionId?: string;
  timestamp: string;
}

export interface TestRunResult extends RunMeta {
  kind: 'unit' | 'integration';
  passed: boolean;
  totalTests: number;
  passedTests: number;
  failedTests: number;
  durationMs: number;
  framework: string;
  failingTests: string[];
}

export interface FileMutationStats {
  filePath: string;
  totalMutants: number;
  killed: number;
  survived: number;
  noCoverage: number;
  mutationScore: number;
}

export interface MutationRunResult extends RunMeta {
  kind: 'mutation';
  tool: string;
  totalMutants: number;
  killed: number;
  survived: number;
  noCoverage: number;
  mutationScore: number;
  runtimeMs: number;
  byFile: FileMutationStats[];
}

export interface PIDComponents {
  p: number;
  i: number;
  d: number;
}

export interface RegionMetrics {
  projectId: string;
  commitSha: string;
  regionId: string;
  mutationScore: number;
  centrality: number;
  triviality: number;
  pid: PIDComponents;
}

export interface RunHistory {
  projectId: string;
  regionId?: string;
  runs: (TestRunResult | MutationRunResult)[];
}
