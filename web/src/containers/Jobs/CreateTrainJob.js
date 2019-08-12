import React from 'react'

import { withStyles } from '@material-ui/core/styles'
import { compose } from 'redux'
import { connect } from 'react-redux'

// Material UI
import { Toolbar, Typography, Grid } from '@material-ui/core'

import * as DatasetActions from "../Datasets/actions"
import * as ConsoleActions from "../ConsoleAppFrame/actions"
import * as actions from "./actions"


// Import Layout
import MainContent from 'components/Console/ConsoleContents/MainContent'
import ContentBar from 'components/Console/ConsoleContents/ContentBar'

import CreateTrainJobForm from "components/Console/ConsoleForms/CreateTrainJobForm";

const styles = theme => ({
    block: {
        display: 'block',
    },
    addDS: {
        marginRight: theme.spacing(1),
    },
    contentWrapper: {
        margin: '40px 16px',
        //position: "relative",
        minHeight: 200,
    },
})

class CreateTrainJob extends React.Component {
    componentDidMount() {
        this.props.requestDatasetsList()
    }

    render() {
        const {
            classes,
            DatasetsList
        } = this.props

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
                    <div className={classes.contentWrapper}>
                        <CreateTrainJobForm datasets={DatasetsList} />
                    </div>
                </MainContent>
            </React.Fragment>
        )
    }
}


const mapStateToProps = state => ({
    DatasetsList: state.DatasetsReducer.DatasetList,
})

const mapDispatchToProps = {
    handleHeaderTitleChange: ConsoleActions.handleHeaderTitleChange,
    postCreateTrainJobs: actions.createTrainJob,
    requestDatasetsList: DatasetActions.requestListDS,
    resetLoadingBar: ConsoleActions.resetLoadingBar,
}

export default compose(
    connect(mapStateToProps, mapDispatchToProps),
    withStyles(styles)
)(CreateTrainJob)