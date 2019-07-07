import * as React from 'react';
import { Route } from "react-router-dom";
import { withStyles, } from '@material-ui/core/styles';
import * as moment from 'moment';

import { Paper, List, ListItem, Typography, Divider, Table, TableBody, TableRow, 
  CircularProgress, ListItemText, TableCell } from '@material-ui/core';
import * as echarts from 'echarts';
import { AppRoute } from '../../app/AppNavigator';

/* interface Props {
  classes: { [s: string]: any };
  appUtils: AppUtils;
  trialId: string;
} */


class TrialDetailPage extends React.Component {

  render() {
    const { classes, appUtils } = this.props;

    return <Route path={AppRoute.TRIAL_DETAIL} render={(props) => {
      const { trialId } = props.match.params;
      return <TrialDetails trialId={trialId} classes={classes} appUtils={appUtils} />
    }} />;
    
  }
}


class TrialDetails extends React.Component {
  charts = [];
  state = { logs: null, trial: null }

  async componentDidMount() {
    const { appUtils: { rafikiClient, showError }, trialId } = this.props;
    try {
      const [logs, trial] = await Promise.all([
        rafikiClient.getTrialLogs(trialId),
        rafikiClient.getTrial(trialId)
      ]);

      this.setState({ logs, trial });
    } catch (error) {
      showError(error, 'Failed to retrieve trial & its logs');
    }
  }

  componentDidUpdate() {
    this.updateCharts();
  }

  updateCharts() {
    const { logs } = this.state;

    if (!logs) return;
    
    this.charts = [];
    const chartOptions = getPlotChartOptions(logs);

    for (const i in chartOptions) {
      const chartOption = chartOptions[i];
      const dom = document.getElementById(`plot-${i}`);

      if (!dom) continue;

      // @ts-ignore
      const chart = echarts.init(dom);  
      chart.setOption(chartOption);
    }
  }

  renderDetails() {
    const { classes } = this.props;
    const { trial } = this.state;

    return (
      <React.Fragment>
        <Typography gutterBottom variant="h3">Details</Typography>
        <Paper className={classes.detailsPaper}>
          <Table padding="dense">
            <TableBody>
              <TableRow>
                <TableCell>ID</TableCell>
                <TableCell>{trial.id}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Model</TableCell>
                <TableCell>{trial.model_name}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Status</TableCell>
                <TableCell>{trial.status}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Score</TableCell>
                <TableCell>{trial.score}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Knobs</TableCell>
                <TableCell>{JSON.stringify(trial.knobs, null, 2)}</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Started at</TableCell>
                <TableCell>{moment(trial.datetime_started).format('llll')}</TableCell>
              </TableRow>
              {
                trial.datetime_stopped &&
                <React.Fragment>
                  <TableRow>
                    <TableCell>Stopped at</TableCell>
                    <TableCell>{moment(trial.datetime_stopped).format('llll')}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Duration</TableCell>
                    <TableCell>{
                      // @ts-ignore
                      moment.duration(trial.datetime_stopped - trial.datetime_started).asMinutes()
                    } min</TableCell>
                  </TableRow>
                </React.Fragment> 
              }
            </TableBody>
          </Table>
        </Paper>
      </React.Fragment>
    );
  }

  renderLogsPlots() {
    const { logs } = this.state;
    const { classes } = this.props;

    return (
      // Show plots section if there are plots
      Object.values(logs.plots).length > 0 &&
      <React.Fragment>
        <Typography gutterBottom variant="h3">Plots</Typography>
        {Object.values(logs.plots).map((x, i) => {
          return <Paper key={x.title} id={`plot-${i}`} className={classes.plotPaper}></Paper>;
        })}
      </React.Fragment>
    )
  }

  renderLogsMessages() {
    const { logs } = this.state;
    const { classes } = this.props;

    return (
      // Show messages section if there are messages
      Object.values(logs.messages).length > 0 &&
      <React.Fragment>
        <Typography gutterBottom variant="h3">Messages</Typography>
        <Paper className={classes.messagesPaper}>
          <List>
            {Object.values(logs.messages).map((x, i) => {
              return (
                <ListItem key={(x.time || '') + x.message}>
                  <ListItemText primary={x.message} secondary={x.time ? x.time.toTimeString() : null} />
                </ListItem>
              );
            })}
            
          </List>
        </Paper>
      </React.Fragment>
    )
  }

  render() {
    const { classes, trialId } = this.props;
    const { logs, trial } = this.state;

    return (
      <React.Fragment>
        <Typography gutterBottom variant="h2">
          Trial
          <span className={classes.headerSub}>{`(ID: ${trialId})`}</span>
        </Typography>
        {
          trial &&
          this.renderDetails()
        }
        {
          logs && (Object.values(logs.plots).length > 0 || Object.values(logs.messages).length > 0) &&
          <Divider className={classes.divider} />
        }
        {
          logs && logs.plots &&
          this.renderLogsPlots()
        }
        {
          logs && Object.values(logs.plots).length > 0 && Object.values(logs.messages).length > 0 &&
          <Divider className={classes.divider} />
        }
        {
          logs && logs.messages &&
          this.renderLogsMessages()
        }
        {
          !(trial && logs) && 
          <CircularProgress />
        }
      </React.Fragment>
    );
  }
}

function getPlotChartOptions(logs) {
  const chartOptions = [];

  for (const plot of logs.plots) {
    const points = [];

    for (const metric of logs.metrics) {
      // Check if x axis value exists
      const xAxis = plot.x_axis || 'time';
      if (!(xAxis in metric)) {
        continue;
      }

      // Initialize point
      const point = [metric[xAxis]];

      // For each of plot's y axis metrics, add it to point data
      for (const plotMetric of plot.metrics) {
        point.push(plotMetric in metric ? metric[plotMetric] : null);
      }

      points.push(point);
    }

    const series = [];
    series.push({
      name: "Radom Name",
      type: 'line',
      data: points
    });

    const chartOption = {
      title: {
        text: plot.title
      },
      tooltip: {
        trigger: 'axis'
      },
      xAxis: {
        type:  plot.x_axis ? 'value' : 'time',
        splitLine: {
          show: false
        }
      },
      yAxis: {
        type: 'value',
        splitLine: {
          show: false
        }
      },
      series
    };

    chartOptions.push(chartOption);
  }

  return chartOptions;
}

const styles = (theme) => ({
  headerSub: {
    fontSize: theme.typography.h4.fontSize,
    margin: theme.spacing.unit * 2
  },
  detailsPaper: {
    margin: theme.spacing.unit * 2
  },
  messagesPaper: {
    margin: theme.spacing.unit * 2
  },
  plotPaper: {
    width: '100%',
    maxWidth: 800,
    height: 500,
    padding: theme.spacing.unit,
    paddingTop: theme.spacing.unit * 2,
    margin: theme.spacing.unit * 4
  },
  divider: {
    margin: theme.spacing.unit * 4
  }
});

export default withStyles(styles)(TrialDetailPage);