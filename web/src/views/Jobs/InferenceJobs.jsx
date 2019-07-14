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
import Icon from "@material-ui/core/Icon";
// core components
import GridItem from "components/Grid/GridItem.jsx";
import GridContainer from "components/Grid/GridContainer.jsx";
import CustomInput from "components/CustomInput/CustomInput.jsx";
import Button from "components/CustomButtons/Button.jsx";
import Card from "components/Card/Card.jsx";
import CardHeader from "components/Card/CardHeader.jsx";
import CardBody from "components/Card/CardBody.jsx";
import CardFooter from "components/Card/CardFooter.jsx";
import FormHelperText from '@material-ui/core/FormHelperText';

const styles = {
  cardCategoryWhite: {
    color: "rgba(255,255,255,.62)",
    margin: "0",
    fontSize: "14px",
    marginTop: "0",
    marginBottom: "0"
  },
  cardTitleWhite: {
    color: "#FFFFFF",
    marginTop: "0px",
    minHeight: "auto",
    fontWeight: "300",
    fontFamily: "'Roboto', 'Helvetica', 'Arial', sans-serif",
    marginBottom: "3px",
    textDecoration: "none"
  }
};

function InferenceJobs(props) {
  const { classes } = props;

  return (
    <div>
      <GridContainer justify="center" alignContent="center">
        <GridItem xs={12} sm={12} md={12} lg={10} xl={8}>
          <Card>
            <CardHeader color="warning">
              <h4 className={classes.cardTitleWhite}><Icon>add_photo_alternate</Icon>InferenceJobs</h4>
              <p className={classes.cardCategoryWhite}>Image Claasification</p>
            </CardHeader>
            <CardBody>
            <GridContainer>
                <GridItem xs={12} sm={12} md={6}>
                  <CustomInput
                    labelText="JobID"
                    id="Jobid"
                    formControlProps={{
                      fullWidth: true
                    }}
                    inputProps={{
                      disabled: true
                    }}
                  />
                </GridItem>
              </GridContainer>
              <GridContainer>
                <GridItem xs={12} sm={12} md={6}>
                  <CustomInput
                    labelText="Application"
                    id="application"
                    formControlProps={{
                      fullWidth: true
                    }}
                  />
                </GridItem>
                <GridItem xs={12} sm={12} md={6}>
                  <CustomInput
                    labelText="App Version"
                    id="appVersion"
                    formControlProps={{
                      fullWidth: true
                    }}
                  />
                </GridItem>
              </GridContainer>
              <GridContainer>
                <GridItem xs={12} sm={12} md={12}>
                  <p className={classes.cardCategory}>Test your train job</p>
                  <label htmlFor="outlined-button-file">
                    <Button variant="outlined" component="span" className={classes.button}>
                      Upload Image
                    </Button>
                  </label>
                  <FormHelperText id="component-helper-text">*The upload image should have the <b>dimensions W x H</b>, accepted formats are <b>JPEG or PNG</b></FormHelperText>
                </GridItem>
              </GridContainer>
              <GridContainer>
                <GridItem xs={12} sm={12} md={12}>
                  <h6 className={classes.cardCategory}>Apply your train job with Python</h6>
                </GridItem>
              </GridContainer>
            </CardBody>
            <CardFooter>
            </CardFooter>
          </Card>
        </GridItem>
      </GridContainer>
    </div>
  );
}

export default withStyles(styles)(InferenceJobs);
