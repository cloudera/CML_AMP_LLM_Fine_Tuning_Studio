from ft.proto.fine_tuning_studio_pb2_grpc import FineTuningStudioServicer


class FineTuningStudioApp(FineTuningStudioServicer):

    def ListDatasets(self, request, context):
        return super().ListDatasets(request, context)

    def GetDataset(self, request, context):
        return super().GetDataset(request, context)

    def AddDataset(self, request, context):
        return super().AddDataset(request, context)

    def RemoveDataset(self, request, context):
        return super().RemoveDataset(request, context)

    def ListModels(self, request, context):
        return super().ListModels(request, context)

    def GetModel(self, request, context):
        return super().GetModel(request, context)

    def AddModel(self, request, context):
        return super().AddModel(request, context)

    def ExportModel(self, request, context):
        return super().ExportModel(request, context)

    def RemoveModel(self, request, context):
        return super().RemoveModel(request, context)

    def ListAdapters(self, request, context):
        return super().ListAdapters(request, context)

    def GetAdapter(self, request, context):
        return super().GetAdapter(request, context)

    def AddAdapter(self, request, context):
        return super().AddAdapter(request, context)

    def RemoveAdapter(self, request, context):
        return super().RemoveAdapter(request, context)

    def ListPrompts(self, request, context):
        return super().ListPrompts(request, context)

    def GetPrompt(self, request, context):
        return super().GetPrompt(request, context)

    def AddPrompt(self, request, context):
        return super().AddPrompt(request, context)

    def RemovePrompt(self, request, context):
        return super().RemovePrompt(request, context)

    def ListFineTuningJobs(self, request, context):
        return super().ListFineTuningJobs(request, context)

    def GetFineTuningJob(self, request, context):
        return super().GetFineTuningJob(request, context)

    def StartFineTuningJob(self, request, context):
        return super().StartFineTuningJob(request, context)

    def RemoveFineTuningJob(self, request, context):
        return super().RemoveFineTuningJob(request, context)

    def ListEvaluationJobs(self, request, context):
        return super().ListEvaluationJobs(request, context)

    def GetEvaluationJob(self, request, context):
        return super().GetEvaluationJob(request, context)

    def StartEvaluationJob(self, request, context):
        return super().StartEvaluationJob(request, context)

    def RemoveEvaluationJob(self, request, context):
        return super().RemoveEvaluationJob(request, context)

    def GetAppState(self, request, context):
        return super().GetAppState(request, context)
