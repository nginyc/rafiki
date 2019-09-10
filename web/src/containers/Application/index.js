import React from 'react';
import PropTypes from 'prop-types';
import { connect } from "react-redux"
import { compose } from "redux"
import { Link } from 'react-router-dom'

import * as ConsoleActions from "../ConsoleAppFrame/actions"

import { withStyles } from '@material-ui/core/styles';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import Grid from '@material-ui/core/Grid';
import Button from '@material-ui/core/Button';

import MainContent from 'components/Console/ConsoleContents/MainContent'
import ContentBar from "components/Console/ConsoleContents/ContentBar"


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


class ApplicationPage extends React.Component {
    static propTypes = {
        classes: PropTypes.object.isRequired,
        handleHeaderTitleChange: PropTypes.func,
        resetLoadingBar: PropTypes.func,
    }

    state = {
        applicationID: "",
    }


    componentDidMount() {
        this.props.handleHeaderTitleChange("Application > SomeApplicatioName")
    }

    componentDidUpdate(prevProps, prevState) {

    }

    componentWillUnmount() {
        this.props.resetLoadingBar()
    }

    render() {
        const {
            classes,
        } = this.props;

        return (
            <React.Fragment>
                <MainContent>
                    <ContentBar>
                        <Toolbar>
                            <Grid container spacing={10} justify="space-between" alignItems="center">
                                <Grid item>
                                    <Typography variant="h5" gutterBottom>
                                        SomeApplicatioName
                  </Typography>
                                </Grid>
                                <Grid item>
                                    <Button
                                        variant="contained"
                                        color="primary"
                                        className={classes.addDS}
                                        component={Link}
                                        to="/console/datasets/upload-datasets"
                                    >
                                        Open the page
                  </Button>
                                </Grid>
                            </Grid>
                        </Toolbar>
                    </ContentBar>
                    <div className={classes.contentWrapper}>
                        <Typography color="textSecondary" align="center">
                            Application Details
                        </Typography>
                        <Typography>
                            {JSON.stringify({
                                'app': 'fashion_mnist_app',
                                'app_version': 1,
                                'datetime_started': 'Mon, 17 Dec 2018 07:25:36 GMT',
                                'datetime_stopped': "None",
                                'id': '09e5040e-2134-411b-855f-793927c80b4b',
                                'predictor_host': '127.0.0.1:30000',
                                'status': 'RUNNING',
                                'train_job_id': 'ec4db479-b9b2-4289-8086-52794ffc71c8',
                                'workers': [{
                                    'datetime_started': 'Mon, 17 Dec 2018 07:25:36 GMT',
                                    'datetime_stopped': "None",
                                    'replicas': 2,
                                    'service_id': '661035bb-3966-46e8-828c-e200960a76c0',
                                    'status': 'RUNNING',
                                    'trial': {
                                        'id': '1b7dc65a-87ae-4d42-9a01-67602115a4a4',
                                        'knobs': {
                                            'batch_size': 32,
                                            'epochs': 3,
                                            'hidden_layer_count': 2,
                                            'hidden_layer_units': 36,
                                            'image_size': 32,
                                            'learning_rate': 0.014650971133579896
                                        },
                                        'model_name': 'TfFeedForward',
                                        'score': 0.8269
                                    }
                                },
                                {
                                    'datetime_started': 'Mon, 17 Dec 2018 07:25:36 GMT',
                                    'datetime_stopped': "None",
                                    'replicas': 2,
                                    'service_id': '6a769007-b18f-4271-b3db-8b60ed5fb545',
                                    'status': 'RUNNING',
                                    'trial': {
                                        'id': '0c1f9184-7b46-4aaf-a581-be62bf3f49bf',
                                        'knobs': { 'criterion': 'entropy', 'max_depth': 4 },
                                        'model_name': 'SkDt',
                                        'score': 0.6686
                                    }
                                }]
                            })}
                        </Typography>
                    </div>
                </MainContent>
            </React.Fragment>
        )
    }
}

const mapStateToProps = state => ({
    DatasetList: state.DatasetsReducer.DatasetList,
})

const mapDispatchToProps = {
    handleHeaderTitleChange: ConsoleActions.handleHeaderTitleChange,
    resetLoadingBar: ConsoleActions.resetLoadingBar,
}

export default compose(
    connect(mapStateToProps, mapDispatchToProps),
    withStyles(styles)
)(ApplicationPage)
