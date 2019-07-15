/*
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at
 
    http://www.apache.org/licenses/LICENSE-2.0
 
  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
 */

import React from "react";
// @material-ui/core components
import withStyles from "@material-ui/core/styles/withStyles";
import { CircularProgress } from '@material-ui/core';
// core components
import GridItem from "components/Grid/GridItem.jsx";
import GridContainer from "components/Grid/GridContainer.jsx";
import Table from "components/Table/Table.jsx";
import TableTrials from "../../components/Table/TableTrials";
import Card from "components/Card/Card.jsx";
import CardHeader from "components/Card/CardHeader.jsx";
import CardBody from "components/Card/CardBody.jsx";
import { AppRoute } from '../../app/AppNavigator';
import { Route } from "react-router-dom";

const styles = {
  cardCategoryWhite: {
    "&,& a,& a:hover,& a:focus": {
      color: "rgba(255,255,255,.62)",
      margin: "0",
      fontSize: "14px",
      marginTop: "0",
      marginBottom: "0"
    },
    "& a,& a:hover,& a:focus": {
      color: "#FFFFFF"
    }
  },
  cardTitleWhite: {
    color: "#FFFFFF",
    marginTop: "0px",
    minHeight: "auto",
    fontWeight: "300",
    fontFamily: "'Roboto', 'Helvetica', 'Arial', sans-serif",
    marginBottom: "3px",
    textDecoration: "none",
    "& small": {
      color: "#777",
      fontSize: "65%",
      fontWeight: "400",
      lineHeight: "1"
    }
  },
  popOver: {
    // margin: "10px",
    // backgroundColor: "rgba(255, 255, 255, 0)", 
  },
  buttonAdd: {
    //float: "right",
    //fontSize: "15px",
    //lineHeight: "15px",
    // padding: "10px",
    backgroundColor: "rgba(255, 255, 255, 0.2)",
    border: "0px",
    //borderRadius: "5px",
    "&:hover": {
      backgroundColor: "rgba(255, 255, 255, 0.5)",
    }
  },
  buttonDetails: {
    //fontSize: "15px",
    //lineHeight: "15px",
    padding: "10px",
    borderRadius: "5px",
    border: "0px",
    backgroundColor: "rgba(255, 255, 255, 0.2)",
    "&:hover": {
      backgroundColor: "rgba(255, 255, 255, 0.5)",
    },
  }
};

class TrialsList extends React.Component {

  render() {
    const { classes, appUtils } = this.props;

    return <Route path={AppRoute.TRAIN_JOB_DETAIL} render={(props) => {
      const { app, appVersion } = props.match.params;
      return <Details app={app} appVersion={appVersion} classes={classes} appUtils={appUtils} />
    }} />;
    
  }
}

class Details extends React.Component {

  async componentDidMount() {
    const { appUtils: { rafikiClient, showError }, app, appVersion } = this.props;
    try {
      const trials = await rafikiClient.getTrialsOfTrainJob(app, appVersion);
      this.setState({ trials });
    } catch (error) {
      showError(error, 'Failed to retrieve trials for train job');
    }
  }

  state = {
    anchorEl: null,
  };

  handleClick = event => {
    this.setState({
      anchorEl: event.currentTarget,
    });
  };

  handleClose = () => {
    this.setState({
      anchorEl: null,
    });
  };

  render() {

    const { classes } = this.props;
    const { app, appVersion, appUtils } = this.props

    return (
      <GridContainer>
        <GridItem xs={12} sm={12} md={12}>
          <Card plain>
            <CardHeader color="primary">
              <h4 className={classes.cardTitleWhite}>Selected Train Job</h4>
            </CardHeader>
            { app !== null && appVersion !== null &&
            (<CardBody>
              <Table
                tableHeaderColor="primary"
                tableHead={["App", "App Version"]}
                tableData={[[app, appVersion]]}
              />
            </CardBody>)
            }
        </Card>
        </GridItem>
        <GridItem xs={12} sm={12} md={12}>
          <Card>
            <CardHeader color="warning">
              <h4 className={classes.cardTitleWhite}>Trials for this Train Job</h4> 
            </CardHeader>
            <CardBody>
              { !this.state.trials ? <CircularProgress /> : 
              <TableTrials
                tableHeaderColor="primary"
                tableHead={["#", "ID", "Model", "Score", "Status", "Start time", "Stop time", "Duration"]}
                tableData={this.state.trials}
                appUtils={appUtils}
              />
              }
            </CardBody>
          </Card>
        </GridItem>
      </GridContainer>)
  }
}

export default withStyles(styles)(TrialsList);
