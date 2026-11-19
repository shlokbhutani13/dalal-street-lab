{{- define "dalal.name" -}}
{{- .Release.Name | trunc 40 | trimSuffix "-" -}}
{{- end -}}

{{- define "dalal.labels" -}}
app.kubernetes.io/name: dalal-street-lab
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Values.image.tag | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{- define "dalal.selector" -}}
app.kubernetes.io/name: dalal-street-lab
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "dalal.image" -}}
{{ .Values.image.repository }}:{{ .Values.image.tag }}
{{- end -}}

{{/* Settings every container gets: the ConfigMap, the Secret and the kill switch. */}}
{{- define "dalal.env" -}}
envFrom:
  - configMapRef:
      name: {{ include "dalal.name" . }}-config
  - secretRef:
      name: {{ .Values.existingSecret }}
{{- end -}}

{{/* Pod template for a one-off job that runs one or more dalal-* commands in order. */}}
{{- define "dalal.jobPod" -}}
{{- $root := index . 0 -}}
{{- $command := index . 1 -}}
{{- $resources := index . 2 -}}
metadata:
  labels:
    {{- include "dalal.selector" $root | nindent 4 }}
spec:
  restartPolicy: Never
  securityContext:
    {{- toYaml $root.Values.podSecurityContext | nindent 4 }}
  containers:
    - name: job
      image: {{ include "dalal.image" $root }}
      imagePullPolicy: {{ $root.Values.image.pullPolicy }}
      command: ["sh", "-c", {{ $command | quote }}]
      {{- include "dalal.env" $root | nindent 6 }}
      env:
        - name: MLFLOW_TRACKING_URI
          value: sqlite:////mlflow/mlflow.db
      volumeMounts:
        - { name: mlflow, mountPath: /mlflow }
        - { name: tmp, mountPath: /tmp }
      securityContext:
        {{- toYaml $root.Values.containerSecurityContext | nindent 8 }}
      resources:
        {{- toYaml $resources | nindent 8 }}
  volumes:
    - name: mlflow
      persistentVolumeClaim:
        claimName: {{ include "dalal.name" $root }}-mlflow
    - name: tmp
      emptyDir: {}
{{- end -}}
