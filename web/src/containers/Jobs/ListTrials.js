import React from 'react'
import PropTypes from 'prop-types'

import { connect } from 'react-redux'
import { compose } from "redux"
import { Link } from 'react-router-dom'
import { push } from 'connected-react-router'

// Material UI
import { Button, Table, Toolbar, Typography, Grid, Tooltip, IconButton, TableHead, TableBody, TableRow, TableCell } from '@material-ui/core'
import { withStyles } from '@material-ui/core/styles'
import { Pageview } from '@material-ui/icons'
import RefreshIcon from '@material-ui/icons/Refresh'

import * as ConsoleActions from "../ConsoleAppFrame/actions"
import * as jobsActions from "./actions"
import * as applicationActions from "../Application/actions"

// Import Layout
import MainContent from 'components/Console/ConsoleContents/MainContent'
import ContentBar from 'components/Console/ConsoleContents/ContentBar'

// Third parts
import * as moment from 'moment';

/* ListJobs are able to view trials and Trial details*/

const styles = theme => ({
    block: {
        display: 'block',
    },
    add: {
        marginRight: theme.spacing(1),
    },
    contentWrapper: {
        margin: '40px 16px',
        //position: "relative",
        minHeight: 200,
    },
    contentTrasparent: {
        background: "#0000"
    }
})

class ListTrials extends React.Component {
    static propTypes = {
        classes: PropTypes.object.isRequired,
        handleHeaderTitleChange: PropTypes.func,
        jobsList: PropTypes.array,
        trialsList: PropTypes.array,
        requestTrialsListOfJob: PropTypes.func,
        resetLoadingBar: PropTypes.func
    }

    state = {
        jobsSelected: "",
    }

    reloadListOfTrials = () => {
        const { app, appVersion } = this.props.match.params
        this.props.requestTrialsListOfJob(app, appVersion)
    }

    componentDidMount() {
        const { app, appVersion } = this.props.match.params
        this.props.handleHeaderTitleChange("Training Jobs > Jobs List")
        this.props.requestTrialsListOfJob(app, appVersion)
    }

    componentDidUpdate(prevProps, prevState) {

    }

    componentWillUnmount() {
        this.props.resetLoadingBar()
    }

    render() {
        const {
            classes,
            jobsList,
            match
        } = this.props

        const { appId, app, appVersion } = match.params
        // eslint-disable-next-line
        const job = jobsList.find(job => job.id == appId) // id & appId might not be same type
        let trialsList = []
        // eslint-disable-next-line
        if (job != undefined) {
            trialsList = job.trials || [] 
        }
        return (
            <React.Fragment>
                <MainContent>
                    <ContentBar>
                        <Toolbar>
                            <Grid container spacing={10} justify="space-between" alignItems="center">
                                <Grid item>
                                    <Typography variant="h5" gutterBottom>
                                        Selected Job
                                    </Typography>
                                </Grid>
                                <Grid item>
                                    <Button
                                        variant="contained"
                                        color="primary"
                                        className={classes.add}
                                        component={Link}
                                        to={`/console/application/${appId}/${app}/${appVersion}/create_inference_job`}
                                    >
                                        Create Inference Job
                                    </Button>
                                </Grid>
                            </Grid>
                        </Toolbar>
                    </ContentBar>
                    <div className={classes.contentTrasparent}>
                        <Table>
                            <TableHead>
                                <TableRow>
                                    <TableCell> ID </TableCell>
                                    <TableCell> App </TableCell>
                                    <TableCell> App Version</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                <TableRow>
                                    <TableCell> {appId}</TableCell>
                                    <TableCell> {app} </TableCell>
                                    <TableCell> {appVersion}</TableCell>
                                </TableRow>
                            </TableBody>
                        </Table>
                    </div>
                </MainContent>
                <MainContent>
                    <ContentBar>
                        <Toolbar>
                            <Grid container spacing={10} justify="space-between" alignItems="center">
                                <Grid item>
                                    <Typography variant="h5" gutterBottom>
                                        Trials
                                    </Typography>
                                </Grid>
                                <Grid item>
                                    <Tooltip title="Reload">
                                        <IconButton
                                            onClick={() => {
                                                this.reloadListOfTrials()
                                            }}
                                        >
                                            <RefreshIcon className={classes.block} color="inherit" />
                                        </IconButton>
                                    </Tooltip>
                                </Grid>
                            </Grid>
                        </Toolbar>
                    </ContentBar>
                    <div className={classes.contentWrapper}>
                        <Typography color="textSecondary" align="center">
                            {trialsList.length === 0
                                ? "You do not have any trials for this job"
                                : "Jobs"
                            }
                        </Typography>
                        <Table>
                            <TableHead>
                                <TableRow>{
                                    [" ", "Model", "Trial No", "Score", "Status", "Started", "Stopped"].map((label) => (<TableCell>{label}</TableCell>))
                                }
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {trialsList.map((x) => {
                                    return (
                                        <TableRow key={x.id} hover>
                                             <TableCell padding="none">
                                                <IconButton onClick={() => {
                                                    const link = "/console/jobs/trials/:trialId"
                                                        .replace(':trialId', x.id)
                                                    this.props.push(link)
                                                }} >
                                                    <Pageview />
                                                </IconButton>
                                            </TableCell>
                                            <TableCell>{x.model_name}</TableCell>
                                            <TableCell>{x.no}</TableCell>
                                            <TableCell>{x.score !== null ? x.score : '-'}</TableCell>
                                            <TableCell>{x.status}</TableCell>
                                            <TableCell>{moment(x.datetime_started).fromNow()}</TableCell>
                                            <TableCell>{x.datetime_stopped ? moment(x.datetime_stopped).fromNow() : '-'}</TableCell>
                                        </TableRow>
                                    )
                                }
                                )}
                            </TableBody>
                        </Table>
                    </div>
                </MainContent>
            </React.Fragment >
        )
    }
}

const mapStateToProps = state => ({
    jobsList: state.JobsReducer.jobsList
})

const mapDispatchToProps = {
    createInferenceJob: applicationActions.postCreateInferenceJob,
    handleHeaderTitleChange: ConsoleActions.handleHeaderTitleChange,
    requestTrialsListOfJob: jobsActions.requestTrialsListOfJob,
    resetLoadingBar: ConsoleActions.resetLoadingBar,
    push: push,
}

export default compose(
    connect(mapStateToProps, mapDispatchToProps),
    withStyles(styles)
)(ListTrials)