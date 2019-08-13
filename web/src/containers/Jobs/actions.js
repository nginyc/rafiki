export const Types = {
   // ==== TRAINJOBS ===== 
   REQUEST_TRAIN_JOBSLIST:"Jobs/request_train_jobslist", // async action is often named with three _ in between
   POST_CREAT_TRAINJOB: "Jobs/post_create_trainjob",
   POPULATE_TRAINJOBSLIST: "Jobs/populate_trainjobslist", // sync action is often named with two _ in between

   // === TRIALS ====
   REQUEST_TRIALSLIST_OFJOB: "Trials/request_trialslist_ofjob",
   POPULATE_TRIALSTOJOBS: "Trials/populate_trialstojobs"
}

/* ====== JOBS ======= */

/* This action create new train job*/
export function createTrainJob(json) {
   return {
      type: Types.POST_CREAT_TRAINJOB,
      json
   }
}

export function requestJobsList() {
   return {
      type: Types.REQUEST_TRAIN_JOBSLIST
   }
}

export function populateJobsList(jobsList) {
   return {
      type: Types.POPULATE_TRAINJOBSLIST,
      jobsList
   }
}

/* ====== TRIALS ======= */

export function requestTrialsListOfJob(app, appVersion) {
   return {
      type: Types.REQUEST_TRIALSLIST_OFJOB,
      app, appVersion
   }
}

export function populateTrialsToJobs(trials, app, appVersion) {
   return {
      type: Types.POPULATE_TRIALSTOJOBS,
      trials,
      app,
      appVersion
   }
}