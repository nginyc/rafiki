import React from 'react'
import PropTypes from 'prop-types'
import { Link } from 'react-router-dom'

import { connect } from 'react-redux'
import { compose } from "redux"
import { push } from 'connected-react-router'

import { withStyles } from '@material-ui/core/styles'
import { Pageview } from '@material-ui/icons'

import * as ConsoleActions from "../ConsoleAppFrame/actions"
import * as jobsActions from "./actions"

// Material UI
import { Table, Toolbar, Typography, Grid, Button, Tooltip, IconButton, TableHead, TableBody, TableRow, TableCell } from '@material-ui/core'
import RefreshIcon from '@material-ui/icons/Refresh'

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
})

class ListJobs extends React.Component {
    static propTypes = {
        classes: PropTypes.object.isRequired,
        handleHeaderTitleChange: PropTypes.func,
        JobsList: PropTypes.array,
        requestJobsList: PropTypes.func,
        resetLoadingBar: PropTypes.func
    }

    state = {
        jobsSelected: "",
    }

    reloadJobs = () => {
        // TODO
    }

    componentDidMount() {
        this.props.handleHeaderTitleChange("Training Jobs > Jobs List")
        this.props.requestJobsList()
    }

    componentDidUpdate(prevProps, prevState) {

    }

    componentWillUnmount() {
        this.props.resetLoadingBar()
    }

    render() {
        const {
            classes,
            JobsList
        } = this.props

        return (
            <React.Fragment>
                <MainContent>
                    <ContentBar>
                        <Toolbar>
                            <Grid container spacing={10} justify="space-between" alignItems="center">
                                <Grid item>
                                    <Typography variant="h5" gutterBottom>
                                        Training Jobs
                                    </Typography>
                                </Grid>
                                <Grid item>
                                    <Button
                                        variant="contained"
                                        color="primary"
                                        className={classes.add}
                                        component={Link}
                                        to="/console/jobs/create-train-job"
                                    >
                                        Create New jobs
                                    </Button>
                                    <Tooltip title="Reload">
                                        <IconButton
                                            onClick={console.log}
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
                            {JobsList.length === 0
                                ? "You do not have any jobs"
                                : "Jobs"
                            }
                        </Typography>
                        <Table>
                            <TableHead>
                                <TableRow>
                                    <TableCell> # </TableCell>
                                    <TableCell> App </TableCell>
                                    <TableCell> App Version</TableCell>
                                    <TableCell> Task </TableCell>
                                    <TableCell> Budget </TableCell>
                                    <TableCell> Started</TableCell>
                                    <TableCell> Stopped </TableCell>
                                    <TableCell> Status </TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {JobsList.map((x) => {
                                    return (
                                        <TableRow key={x.id} hover>
                                            <TableCell padding="none">
                                                <IconButton onClick={() => {
                                                    const link = "/console/jobs/trials/:appId/:app/:appVersion"
                                                        .replace(':appId', x.id)
                                                        .replace(':app', x.app)
                                                        .replace(':appVersion', x.app_version);
                                                    this.props.push(link)
                                                }} >
                                                    <Pageview />
                                                </IconButton>
                                            </TableCell>
                                            <TableCell>{x.app}</TableCell>
                                            <TableCell>{x.app_version}</TableCell>
                                            <TableCell>{x.task}</TableCell>
                                            <TableCell>{JSON.stringify(x.budget)}</TableCell>
                                            <TableCell>{moment(x.datetime_started).fromNow()}</TableCell>
                                            <TableCell>{x.datetime_stopped ? moment(x.datetime_stopped).fromNow() : '-'}</TableCell>
                                            <TableCell>{x.status}</TableCell>
                                        </TableRow>
                                    )}
                                )}
                            </TableBody>
                        </Table>
                    </div>
                </MainContent>
            </React.Fragment>
        )
    }
}

const mapStateToProps = state => ({
    JobsList: state.JobsReducer.jobsList
})

const mapDispatchToProps = {
    handleHeaderTitleChange: ConsoleActions.handleHeaderTitleChange,
    requestJobsList: jobsActions.requestJobsList,
    resetLoadingBar: ConsoleActions.resetLoadingBar,
    push: push
}

export default compose(
    connect(mapStateToProps, mapDispatchToProps),
    withStyles(styles)
)(ListJobs)