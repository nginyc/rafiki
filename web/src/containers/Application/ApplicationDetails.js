import React from 'react'

import { withStyles } from '@material-ui/core/styles'
import { compose } from 'redux'
import { connect } from 'react-redux'

import { goBack } from "connected-react-router"

// Material UI
import { Button, Table, Toolbar, Typography, Grid, TableHead, TableBody, TableRow, TableCell } from '@material-ui/core'

import * as ConsoleActions from "../ConsoleAppFrame/actions"
import * as ClientAPI from "services/ClientAPI"

// Third parts
import * as moment from 'moment';

// Import Layout
import MainContent from 'components/Console/ConsoleContents/MainContent'
import ContentBar from 'components/Console/ConsoleContents/ContentBar'

const styles = theme => ({
    block: {
        display: 'block',
    },
    addDS: {
        marginRight: theme.spacing(1),
    },
    contentWrapper: {
        margin: '40px 16px 0px 16px',
        //position: "relative",
        textAlign: "center",
    },
})

class ApplicationDetails extends React.Component {
    state = { selectedApplication: {} }

    async componentDidMount() {
        this.props.handleHeaderTitleChange("Application > List Application")
        const { app, appVersion } = this.props.match.params
        const { token } = this.props
        const response = await ClientAPI.get_running_inference_jobs(app, appVersion, {}, token)
        const application = response.data
        this.setState({ selectedApplication: application })
    }

    render() {
        const {
            classes,
        } = this.props

        const { app, appVersion } = this.props.match.params
        const x = this.state.selectedApplication

        return (
            <React.Fragment>
                <MainContent>
                    <ContentBar>
                        <Toolbar>
                            <Grid container spacing={10} justify="space-between" alignItems="center">
                                <Grid item>
                                    <Typography variant="h5" gutterBottom>
                                        Create Train Jobs
                                    </Typography>
                                </Grid>
                            </Grid>
                        </Toolbar>
                    </ContentBar>
                    <Grid container spacing={10} justify="center" alignItems="center">
                        <Grid item xs={12} justify="center" alignItems="center">
                            <div className={classes.contentWrapper} >
                                <p>
                                    List of running inference jobs for <b>{app}</b> | appVersion: <b>{appVersion}</b>
                                </p>
                            </div>
                        </Grid>
                        <Grid item>
                            <Table>
                                <TableHead>
                                    <TableRow>{
                                        ["ID", "App", "App Version",  "Started", "Prediction Host"].map((label) => (<TableCell>{label}</TableCell>))
                                    }
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {x ?
                                        <TableRow key={x.id} hover>
                                            <TableCell>{x.id}</TableCell>
                                            <TableCell>{x.app}</TableCell>
                                            <TableCell>{x.app_version}</TableCell>
                                            <TableCell>{moment(x.datetime_started).fromNow()}</TableCell>
                                            <TableCell>{x.predictor_host}</TableCell>
                                        </TableRow> : ""
                                    }
                                </TableBody>
                            </Table>
                        </Grid>
                    </Grid>
                    <Grid container spacing={5} justify="center" alignItems="center" style={{ minHeight: "100px" }}>
                        {/* <Grid item >
                            <Button onClick={this.onClick} color="primary" variant="contained">
                                Stop Inference Job
                            </Button>
                        </Grid> */}
                        <Grid item>
                            <Button color="default" variant="contained" onClick={this.props.goBack}>
                                Go Back
                            </Button>
                        </Grid>
                    </Grid>
                </MainContent>
            </React.Fragment >
        )
    }
}


const mapStateToProps = state => ({
    // ApplicationList: state.ApplicationReducer.JobsList,
    token: state.Root.token,
})

const mapDispatchToProps = {
    handleHeaderTitleChange: ConsoleActions.handleHeaderTitleChange,
    resetLoadingBar: ConsoleActions.resetLoadingBar,
    goBack: goBack
}

export default compose(
    connect(mapStateToProps, mapDispatchToProps),
    withStyles(styles)
)(ApplicationDetails)