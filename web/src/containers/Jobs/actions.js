export const Types = {
   REQUEST_TRAIN_JOBSLIST:"Jobs/request_train_jobslist",
   POST_CREAT_TRAINJOB: "Jobs/post_create_trainjob",
   POPULATE_TRAINJOBSLIST: "Jobs/populate_trainjobslist"
}

export function requestJobsList() {
   return {
      type: Types.REQUEST_TRAIN_JOBSLIST
   }
}

export function populateTrainJobslist(jobsList) {
   return {
      type: Types.POPULATE_TRAINJOBSLIST,
      jobsList
   }
}