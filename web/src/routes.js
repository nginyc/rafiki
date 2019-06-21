// @material-ui/icons
import Dashboard from "@material-ui/icons/Dashboard";

/* // core components/views for Admin layout
import DashboardPage from "views/Dashboard/Dashboard.jsx";
*/

import Jobs from "./views/Jobs/Jobs";
import TrainJobs from "views/Jobs/TrainJobs.jsx"
import InferenceJobs from "./views/Jobs/InferenceJobs"

import Trials from "./views/Trials/Trials";
import TrialsDetail from "./views/Trials/TrialsDetail";

import Datasets from "./views/Datasets/Datasets"
import NewDataset from "./views/Datasets/NewDataset";

// core components/views for RTL layout

const dashboardRoutes = [
  {
    path: "/datasets/new",
    name: "New Dataset",
    rtlName: "قائمة الجدول",
    icon: "playlist_add",
    component: NewDataset,
    layout: "/admin"
  }, 
 /* {
    path: "/datasets/details",
    name: "Datasets Details",
    rtlName: "قائمة الجدول",
    icon: "playlist_add",
    component: DashboardPage,
    layout: "/admin"
  }, */
  {
    path: "/datasets",
    name: "Datasets",
    rtlName: "لوحة القيادة",
    icon: "recent_actors",
    component: Datasets,
    layout: "/admin"
  },
  {
    path: "/jobs/trials/deploy",
    name: "Deploy trials",
    rtlName: "قائمة الجدول",
    icon: "playlist_play",
    component: InferenceJobs,
    layout: "/admin"
  },
  {
    path: "/jobs/new",
    name: "Train Jobs",
    rtlName: "قائمة الجدول",
    icon: "playlist_add",
    component: TrainJobs,
    layout: "/admin"
  },
  {
    path: "/jobs/trials/details",
    name: "Trial Details",
    rtlName: "قائمة الجدول",
    icon: Dashboard,
    component: TrialsDetail,
    layout: "/admin"
  },
  {
    path: "/jobs/trials",
    name: "Trials",
    rtlName: "قائمة الجدول",
    icon: "playlist_play",
    component: Trials,
    layout: "/admin"
  },
  {
    path: "/jobs",
    name: "Jobs",
    rtlName: "قائمة الجدول",
    icon: "playlist_add",
    component: Jobs,
    layout: "/admin"
  },
  {
    path: "/application",
    name: "Inference Jobs",
    rtlName: "قائمة الجدول",
    icon: "playlist_add_check",
    component: InferenceJobs,
    layout: "/admin"
  },
  /*{
    path: "/dashboard",
    name: "Dashboard",
    rtlName: "لوحة القيادة",
    icon: Dashboard,
    component: DashboardPage,
    layout: "/admin"
  }
  {
    path: "/user",
    name: "User Profile",
    rtlName: "ملف تعريفي للمستخدم",
    icon: Person,
    component: UserProfile,
    layout: "/admin"
  },
  {
    path: "/table",
    name: "Table List",
    rtlName: "قائمة الجدول",
    icon: "content_paste",
    component: TableList,
    layout: "/admin"
  },
  {
    path: "/typography",
    name: "Typography",
    rtlName: "طباعة",
    icon: LibraryBooks,
    component: Typography,
    layout: "/admin"
  },
  {
    path: "/icons",
    name: "Icons",
    rtlName: "الرموز",
    icon: BubbleChart,
    component: Icons,
    layout: "/admin"
  },
  {
    path: "/maps",
    name: "Maps",
    rtlName: "خرائط",
    icon: LocationOn,
    component: Maps,
    layout: "/admin"
  },
  {
    path: "/notifications",
    name: "Notifications",
    rtlName: "إخطارات",
    icon: Notifications,
    component: NotificationsPage,
    layout: "/admin"
  },
  {
    path: "/upgrade-to-pro",
    name: "Upgrade To PRO",
    rtlName: "التطور للاحترافية",
    icon: Unarchive,
    component: UpgradeToPro,
    layout: "/admin"
  },
  {
    path: "/rtl-page",
    name: "RTL Support",
    rtlName: "پشتیبانی از راست به چپ",
    icon: Language,
    component: RTLPage,
    layout: "/rtl"
  } */
];

export default dashboardRoutes;
